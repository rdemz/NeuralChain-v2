// Export des modules
pub mod block;
pub mod blockchain;
pub mod continuous_mining;
pub mod crypto;
pub mod transaction;
pub mod utils;
pub mod wallet;

// Re-export des structures principales
pub use block::Block;
pub use blockchain::{Blockchain, BlockchainState};
pub use transaction::{Transaction, TransactionType, SignatureScheme};
pub use wallet::Wallet;
