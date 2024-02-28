//! Implements a `wasi-nn` [`BackendInner`] using Llama.cpp.

use super::{BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner};
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor};
use crate::{ExecutionContext, Graph};
use serde_json::Value;
use std::path::PathBuf;
use tokio::sync::Mutex;

use llm_chain::options::OptionsCascade;
use llm_chain::prompt::Data;
use llm_chain_llama::context::LLamaContext;
use llm_chain_llama::context::LlamaBatch;
use llm_chain_llama::options::LlamaInvocation;
use llm_chain_llama::options::DEFAULT_OPTIONS;
use llm_chain_llama::tokenizer::tokenize;
use llm_chain_llama::{ContextParams, ModelParams};
use std::sync::Arc;

#[derive(Default)]
pub struct LlamaCppBackend {}

unsafe impl Send for LlamaCppBackend {}
unsafe impl Sync for LlamaCppBackend {}

impl BackendInner for LlamaCppBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Ggml
    }

    fn load(
        &mut self,
        builders: &[&[u8]],
        _target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        if builders.len() >= 2 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
        }

        let model_path: PathBuf = match builders.get(0) {
            Some(bytes) => {
                let path_str = std::str::from_utf8(bytes).unwrap();
                PathBuf::from(path_str)
            }
            None => {
                return Err(BackendError::BackendAccess(anyhow::anyhow!(
                    "Could not parse first builder element into a Path"
                )))
            }
        };
        println!("LlamaCppBackend: model_path: {:?}", model_path);
        let box_: Box<dyn BackendGraph> = Box::new(LlamaCppGraph { model_path });
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        None
    }
}

struct LlamaCppGraph {
    model_path: PathBuf,
}

unsafe impl Send for LlamaCppGraph {}
unsafe impl Sync for LlamaCppGraph {}

impl BackendGraph for LlamaCppGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let model_path = &self.model_path;
        let model_params = ModelParams::default();
        let context_params = ContextParams::default();

        let context = Arc::new(Mutex::new(
            LLamaContext::from_file_and_params(
                model_path.to_str().unwrap(),
                Some(&model_params),
                Some(&context_params),
            )
            .unwrap(),
        ));

        Ok(ExecutionContext(Box::new(LlamaCppExecutionContext {
            context,
            context_params,
            prompt: None,
            output: Arc::new(Mutex::new(Vec::<String>::new())),
        })))
    }
}

struct LlamaCppExecutionContext {
    pub(crate) context: Arc<Mutex<LLamaContext>>,
    pub(crate) context_params: ContextParams,
    prompt: Option<String>,
    output: Arc<Mutex<Vec<String>>>,
}
//unsafe impl Send for LlamaCppExecutionContext {}
//unsafe impl Sync for LlamaCppExecutionContext {}

impl BackendExecutionContext for LlamaCppExecutionContext {
    fn set_input(&mut self, index: u32, tensor: &Tensor) -> Result<(), BackendError> {
        if index == 0 {
            // TODO: use errors instead of unwrap
            let _json: Value = serde_json::from_slice(&tensor.data).unwrap();
            let _model_params = ModelParams::default();
        } else {
            // TODO: use errors instead of unwrap
            self.prompt = Some(String::from_utf8(tensor.data.clone()).unwrap());
        }
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        use tokio::runtime::Builder;

        let prompt = self.prompt.as_ref().unwrap().clone();
        let context_size = self.context_params.n_ctx as usize;
        let context = self.context.clone();
        let options = &DEFAULT_OPTIONS;
        let oc = OptionsCascade::new();
        let oc = oc.with_options(options);
        let llama_invocation = LlamaInvocation::new(oc, &Data::Text(prompt.to_string())).unwrap();

        let output = self.output.clone();

        let rt = Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let _ = rt.spawn_blocking(move || {
            let context = context.blocking_lock();
            let _token_eos = context.llama_token_eos();

            context.llama_kv_cache_clear();

            let tokenized_stop_prompt = tokenize(
                &context,
                llama_invocation
                    .stop_sequence
                    .first() // XXX: Handle multiple stop seqs
                    .map(|x| x.as_str())
                    .unwrap_or("\n\n"),
                false,
                true,
            );

            if tokenized_stop_prompt.len() > context_size {
                println!("Context too small {}", context_size);
                //return;
            }

            let tokenized_input = tokenize(&context, prompt.as_str(), true, false);
            if tokenized_input.len() > context_size {
                eprintln!("Input too long, truncating to {} tokens", context_size);
            }

            let mut embd = tokenized_input.clone();

            let mut batch = LlamaBatch::new_with_tokens(tokenized_input.clone(), 1);
            let last_idx = batch.token_count() - 1;
            batch.enable_logits(last_idx);

            context.llama_decode(&batch).unwrap();
            let mut n_cur = batch.token_count();
            let mut n_used = batch.token_count() - 1;

            let mut n_remaining = context_size - tokenized_input.len();

            embd.resize(context_size, 0);
            let token_eos = context.llama_token_eos();

            let mut stop_sequence_i = 0;
            let mut n_batch = batch.token_count();
            let mut n_samples = 0;
            let ignore_initial_nls = prompt.ends_with('?');
            let nl_token = context.llama_token_nl();

            while n_remaining > 0 {
                let tok = context.llama_sample(
                    context_size as i32,
                    embd.as_slice(),
                    n_used as i32,
                    &llama_invocation,
                    n_batch as i32,
                );
                n_samples += 1;
                n_used += 1;
                n_remaining -= 1;
                embd[n_used] = tok;

                if tok == token_eos {
                    break;
                }
                if llama_invocation.n_tok_predict != 0
                    && n_used > llama_invocation.n_tok_predict + tokenized_input.len() - 1
                {
                    break;
                }
                // If the input prompt is in the form of a question then next
                // predicted tok will be a new line to finish off the question
                // itself, followed by another new line before the actual
                // answer. This is what the following is checking for.
                if n_samples <= 2 && ignore_initial_nls && tok == nl_token {
                    continue;
                }
                if tok == tokenized_stop_prompt[stop_sequence_i] {
                    stop_sequence_i += 1;
                    if stop_sequence_i >= tokenized_stop_prompt.len() {
                        break;
                    }
                } else {
                    let piece = context.llama_token_to_piece(tok).unwrap();
                    //use std::io::Write;
                    //print!("{}", &piece);
                    // flush stdout after each token
                    //std::io::stdout().flush().unwrap();

                    let mut output = output.blocking_lock();
                    output.push(piece.to_string());

                    stop_sequence_i = 0;

                    let batch = LlamaBatch::new_with_token(tok, n_cur as i32);

                    n_batch = batch.token_count();
                    n_cur += 1;

                    context.llama_decode(&batch).unwrap();
                }
            }
        });
        Ok(())
    }

    fn get_output(&mut self, _index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        let output = self.output.blocking_lock();
        let bytes: Vec<u8> = output.iter().flat_map(|s| s.as_bytes()).cloned().collect();
        if bytes.len() > destination.len() {
            return Err(BackendError::NotEnoughMemory(bytes.len()));
        }
        destination[..bytes.len()].copy_from_slice(&bytes);
        Ok(bytes.len() as u32)
    }
}

// Test can be run using the following command:
// cargo t --features="llama_cpp" backend::llama_cpp::tests::test_load -- --exact --show-output
//
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_init_execution_context() {
        let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tests_dir = Path::new(&project_root).join("tests");
        // https://huggingface.co/Havmand/minillama
        let model_path = tests_dir.join("minillama.gguf");

        let mut backend = LlamaCppBackend {};
        let builders: &[&[u8]] = &[&model_path
            .to_str()
            .expect("Path should contain valid UTF-8")
            .as_bytes()];
        let graph = backend.load(builders, ExecutionTarget::Cpu).unwrap();
        let result = graph.init_execution_context();
        assert!(result.is_ok());
        let mut execution_context = result.unwrap();
        let llama_cpp_exec_context = execution_context.0;
    }

    #[tokio::test]
    async fn test_set_input() {
        let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tests_dir = Path::new(&project_root).join("tests");
        // https://huggingface.co/Havmand/minillama
        let model_path = tests_dir.join("llama-2-7b.Q4_0.gguf");

        let mut backend = LlamaCppBackend {};
        let builders: &[&[u8]] = &[&model_path
            .to_str()
            .expect("Path should contain valid UTF-8")
            .as_bytes()];
        let graph = backend.load(builders, ExecutionTarget::Cpu).unwrap();
        let result = graph.init_execution_context();
        assert!(result.is_ok());
        let mut execution_context = result.unwrap();
        let mut backend_exec_context = execution_context.0;
        // Options consist of options for the model, for the context, for
        // sampling, and also for the wasmtime-wasi-nn (like logging perhaps).
        let options = serde_json::json!({
            "stream-stdout": true,
            "enable-log": true,
            "ctx-size": 1024,
            "n-predict": 512,
            "n-gpu-layers": 25
        });
        let options_tensor = Tensor {
            dimensions: vec![1_u32],
            tensor_type: TensorType::U8,
            data: options.to_string().as_bytes().to_vec(),
        };
        backend_exec_context.set_input(1, &options_tensor);

        let prompt = "Somewhere over ";
        let prompt_tensor = Tensor {
            dimensions: vec![1_u32],
            tensor_type: TensorType::U8,
            data: prompt.to_string().as_bytes().to_vec(),
        };
        backend_exec_context.set_input(2, &prompt_tensor);

        let result = backend_exec_context.compute();
    }

    #[test]
    fn test_load() {
        let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tests_dir = Path::new(&project_root).join("tests");
        // https://huggingface.co/Havmand/minillama
        let model_path = tests_dir.join("minillama.gguf");

        let mut backend = LlamaCppBackend {};
        let builders: &[&[u8]] = &[&model_path
            .to_str()
            .expect("Path should contain valid UTF-8")
            .as_bytes()];
        let result = backend.load(builders, ExecutionTarget::Cpu);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_with_invalid_number_of_builders() {
        let mut backend = LlamaCppBackend {};
        let builders: &[&[u8]] = &[&[1, 2, 3], &[4, 5, 6]];
        let result = backend.load(builders, ExecutionTarget::Cpu);
        assert!(matches!(
            result,
            Err(BackendError::InvalidNumberOfBuilders(1, 2))
        ));
    }
}
// llama_model_params:
// n_gpu_layers
// split mode
// main_gpu
// tensor_split
// vocab_only
// use_mmap
// use_mlock
//
// llama_context_params:
// seed;              // RNG seed, -1 for random
// n_ctx;             // text context, 0 = from model
// n_batch;           // prompt processing maximum batch size
// n_threads;         // number of threads to use for generation
// n_threads_batch;   // number of threads to use for batch processing
// rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
// rope_freq_base;   // RoPE base frequency, 0 = from model
// rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
// yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
// yarn_attn_factor; // YaRN magnitude scaling factor
// yarn_beta_fast;   // YaRN low correction dim
// yarn_beta_slow;   // YaRN high correction dim
// yarn_orig_ctx;    // YaRN original context size
// defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)
// embedding;   // embedding mode only
// offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
// do_pooling;  // whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
//
