pub mod block;
pub mod blockchain;
pub mod continuous_mining;
pub mod crypto;
pub mod transaction;
pub mod utils;
pub mod wallet;

// Re-exporte les structures principales
pub use block::Block;
pub use blockchain::Blockchain;
pub use continuous_mining::ContinuousMining;
pub use wallet::Wallet;
