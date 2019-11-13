(function() {var implementors = {};
implementors["wasmtime_api"] = [{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"struct\" href=\"wasmtime_api/struct.Trap.html\" title=\"struct wasmtime_api::Trap\">Trap</a>",synthetic:false,types:["wasmtime_api::trap::Trap"]},];
implementors["wasmtime_environ"] = [{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"enum\" href=\"wasmtime_environ/enum.CompileError.html\" title=\"enum wasmtime_environ::CompileError\">CompileError</a>",synthetic:false,types:["wasmtime_environ::compilation::CompileError"]},];
implementors["wasmtime_jit"] = [{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"enum\" href=\"wasmtime_jit/enum.ActionError.html\" title=\"enum wasmtime_jit::ActionError\">ActionError</a>",synthetic:false,types:["wasmtime_jit::action::ActionError"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"struct\" href=\"wasmtime_jit/struct.UnknownInstance.html\" title=\"struct wasmtime_jit::UnknownInstance\">UnknownInstance</a>",synthetic:false,types:["wasmtime_jit::context::UnknownInstance"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"enum\" href=\"wasmtime_jit/enum.ContextError.html\" title=\"enum wasmtime_jit::ContextError\">ContextError</a>",synthetic:false,types:["wasmtime_jit::context::ContextError"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"enum\" href=\"wasmtime_jit/enum.SetupError.html\" title=\"enum wasmtime_jit::SetupError\">SetupError</a>",synthetic:false,types:["wasmtime_jit::instantiate::SetupError"]},];
implementors["wasmtime_runtime"] = [{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"struct\" href=\"wasmtime_runtime/struct.LinkError.html\" title=\"struct wasmtime_runtime::LinkError\">LinkError</a>",synthetic:false,types:["wasmtime_runtime::instance::LinkError"]},{text:"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/std/error/trait.Error.html\" title=\"trait std::error::Error\">Error</a> for <a class=\"enum\" href=\"wasmtime_runtime/enum.InstantiationError.html\" title=\"enum wasmtime_runtime::InstantiationError\">InstantiationError</a>",synthetic:false,types:["wasmtime_runtime::instance::InstantiationError"]},];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        })()