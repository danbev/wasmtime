//! Runtime library support for Wasmtime.

#![allow(improper_ctypes)]
#![deny(missing_docs, trivial_numeric_casts, unused_extern_crates)]
#![warn(unused_import_braces)]
#![cfg_attr(feature = "std", deny(unstable_features))]
#![cfg_attr(feature = "clippy", plugin(clippy(conf_file = "../../clippy.toml")))]
#![cfg_attr(
    feature = "cargo-clippy",
    allow(clippy::new_without_default, clippy::new_without_default)
)]
#![cfg_attr(
    feature = "cargo-clippy",
    warn(
        clippy::float_arithmetic,
        clippy::mut_mut,
        clippy::nonminimal_bool,
        clippy::option_map_unwrap_or,
        clippy::option_map_unwrap_or_else,
        clippy::print_stdout,
        clippy::unicode_not_nfc,
        clippy::use_self
    )
)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod export;
mod imports;
mod instance;
mod jit_int;
mod memory;
mod mmap;
mod sig_registry;
mod signalhandlers;
mod table;
mod trap_registry;
mod traphandlers;
mod vmcontext;

pub mod libcalls;

pub use crate::export::Export;
pub use crate::imports::Imports;
pub use crate::instance::{InstanceHandle, InstantiationError, LinkError};
pub use crate::jit_int::GdbJitImageRegistration;
pub use crate::mmap::Mmap;
pub use crate::sig_registry::SignatureRegistry;
pub use crate::signalhandlers::{wasmtime_init_eager, wasmtime_init_finish};
pub use crate::trap_registry::{get_mut_trap_registry, get_trap_registry, TrapRegistrationGuard};
pub use crate::traphandlers::{wasmtime_call, wasmtime_call_trampoline};
pub use crate::vmcontext::{
    VMCallerCheckedAnyfunc, VMContext, VMFunctionBody, VMFunctionImport, VMGlobalDefinition,
    VMGlobalImport, VMInvokeArgument, VMMemoryDefinition, VMMemoryImport, VMSharedSignatureIndex,
    VMTableDefinition, VMTableImport,
};

#[cfg(not(feature = "std"))]
use hashbrown::{hash_map, HashMap, HashSet};
#[cfg(feature = "std")]
use std::collections::{hash_map, HashMap, HashSet};

#[cfg(not(feature = "std"))]
use spin::{RwLock, RwLockReadGuard, RwLockWriteGuard};
#[cfg(feature = "std")]
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
