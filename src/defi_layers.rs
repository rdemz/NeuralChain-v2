use std::collections::{HashMap, VecDeque};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use tokio::sync::Mutex;
use std::sync::Arc;

use crate::block::Block;
use crate::blockchain::Blockchain;
use crate::transaction::Transaction;

// Identifiants pour les tokens et les paires
pub type TokenId = [u8; 32];
pub type ChainId = u32;
pub type Address = [u8; 32];
pub type PoolId = Uuid;

// Structure principale du système DeFi multicouche
pub struct DeFiSystem {
    liquidity_pools: HashMap<PoolPair, LiquidityPool>,
    lending_platforms: HashMap<Uuid, LendingPlatform>,
    derivative_markets: HashMap<Uuid, DerivativeMarket>,
    token_bridges: HashMap<ChainId, TokenBridge>,
    oracle_providers: DashMap<Uuid, Arc<OracleProvider>>,
}

// Paires de liquidité
#[derive(Eq, PartialEq, Hash, Clone, Debug, Serialize, Deserialize)]
pub struct PoolPair {
    token_a: TokenId,
    token_b: TokenId,
}

// Pool de liquidité avec swaps automatiques
pub struct LiquidityPool {
    id: PoolId,
    pair: PoolPair,
    reserve_a: Decimal,
    reserve_b: Decimal,
    fee_percent: Decimal,
    lp_tokens: HashMap<Address, Decimal>,
    total_lp_supply: Decimal,
    price_history: VecDeque<PricePoint>,
    created_at: DateTime<Utc>,
    last_swap: Option<DateTime<Utc>>,
}

// Point de prix historique
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PricePoint {
    timestamp: DateTime<Utc>,
    price: Decimal,
    volume: Option<Decimal>,
}

// Plateforme de prêt
pub struct LendingPlatform {
    id: Uuid,
    pools: HashMap<TokenId, LendingPool>,
    collateral_ratios: HashMap<TokenId, Decimal>,
    liquidation_threshold: Decimal,
    oracle: Arc<OracleProvider>,
}

// Pool de prêt pour un token spécifique
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LendingPool {
    token: TokenId,
    total_supplied: Decimal,
    total_borrowed: Decimal,
    utilization_rate: Decimal,
    supply_apy: Decimal,
    borrow_apy: Decimal,
    suppliers: HashMap<Address, Supply>,
    borrowers: HashMap<Address, Borrow>,
}

// Position de fourniture dans un pool de prêt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Supply {
    address: Address,
    amount: Decimal,
    timestamp: DateTime<Utc>,
}

// Position d'emprunt dans un pool de prêt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Borrow {
    address: Address,
    amount: Decimal,
    collateral_token: TokenId,
    collateral_amount: Decimal,
    timestamp: DateTime<Utc>,
    last_interest_update: DateTime<Utc>,
}

// Marché de dérivés
pub struct DerivativeMarket {
    base_asset: TokenId,
    contracts: HashMap<Uuid, DerivativeContract>,
    open_positions: HashMap<Address, Vec<Position>>,
    funding_rate: Decimal,
    next_funding_time: DateTime<Utc>,
}

// Contrat dérivé
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DerivativeContract {
    id: Uuid,
    contract_type: DerivativeType,
    underlying: TokenId,
    strike_price: Option<Decimal>,
    expiry: Option<DateTime<Utc>>,
    oracle: Arc<OracleProvider>,
}

// Types de dérivés supportés
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DerivativeType {
    Perpetual,
    Future { settlement_date: DateTime<Utc> },
    Option { option_type: OptionType, strike_price: Decimal },
    Swap { fixed_rate: Decimal, floating_index: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

// Position ouverte sur un dérivé
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    contract_id: Uuid,
    direction: PositionDirection,
    size: Decimal,
    entry_price: Decimal,
    leverage: Decimal,
    margin: Decimal,
    liquidation_price: Decimal,
    unrealized_pnl: Decimal,
    last_update: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PositionDirection {
    Long,
    Short,
}

// Pont inter-chaînes
pub struct TokenBridge {
    source_chain: ChainId,
    target_chain: ChainId,
    supported_tokens: HashMap<TokenId, BridgeConfig>,
    pending_transfers: HashMap<Uuid, BridgeTransfer>,
}

// Configuration de pont pour un token
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeConfig {
    token_id: TokenId,
    target_token_id: TokenId,
    min_transfer: Decimal,
    max_transfer: Decimal,
    fee_percentage: Decimal,
    validators_required: u8,
    is_active: bool,
}

// Transfert en cours via le pont
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeTransfer {
    id: Uuid,
    token_id: TokenId,
    amount: Decimal,
    sender: Address,
    recipient: Address,
    source_chain: ChainId,
    target_chain: ChainId,
    status: BridgeTransferStatus,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BridgeTransferStatus {
    Pending,
    ValidatorConfirming { confirmations: u8, required: u8 },
    Completed,
    Failed { reason: String },
}

// Fournisseur d'oracle pour les prix
pub struct OracleProvider {
    id: Uuid,
    supported_pairs: HashMap<(TokenId, TokenId), PriceSource>,
    latest_prices: DashMap<(TokenId, TokenId), PriceData>,
    update_interval: chrono::Duration,
}

// Source de prix pour l'oracle
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PriceSource {
    CentralizedExchange { name: String, endpoint: String },
    ChainlinkFeed { address: String, network: String },
    InternalAMM { pool_id: PoolId },
}

// Données de prix pour l'oracle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriceData {
    price: Decimal,
    timestamp: DateTime<Utc>,
    source: PriceSource,
}

#[derive(Error, Debug)]
pub enum DeFiError {
    #[error("Liquidité insuffisante")]
    InsufficientLiquidity,
    
    #[error("Liquidité déséquilibrée")]
    ImbalancedLiquidity,
    
    #[error("Quantité de tokens insuffisante")]
    InsufficientTokens,
    
    #[error("Montant de sortie insuffisant")]
    InsufficientOutputAmount,
    
    #[error("Token invalide")]
    InvalidToken,
    
    #[error("Paire non supportée")]
    UnsupportedPair,
    
    #[error("Prix périmé")]
    StalePrice,
    
    #[error("Tokens LP insuffisants")]
    InsufficientLPTokens,
    
    #[error("Ratio de collatéral insuffisant")]
    InsufficientCollateral,
    
    #[error("Position déjà liquidée")]
    PositionLiquidated,
    
    #[error("Erreur de l'oracle: {0}")]
    OracleError(String),
    
    #[error("Erreur de transaction: {0}")]
    TransactionError(String),
}

impl DeFiSystem {
    pub fn new() -> Self {
        Self {
            liquidity_pools: HashMap::new(),
            lending_platforms: HashMap::new(),
            derivative_markets: HashMap::new(),
            token_bridges: HashMap::new(),
            oracle_providers: DashMap::new(),
        }
    }
    
    // ====== GESTION DES POOLS DE LIQUIDITÉ ======
    
    // Créer un nouveau pool de liquidité
    pub fn create_liquidity_pool(&mut self, token_a: TokenId, token_b: TokenId, fee_percent: Decimal) -> Result<PoolId, DeFiError> {
        // S'assurer que les tokens sont différents
        if token_a == token_b {
            return Err(DeFiError::InvalidToken);
        }
        
        // Créer une paire ordonnée (le plus petit token d'abord)
        let pair = if token_a < token_b {
            PoolPair { token_a, token_b }
        } else {
            PoolPair { token_a: token_b, token_b: token_a }
        };
        
        // Vérifier si le pool existe déjà
        if self.liquidity_pools.contains_key(&pair) {
            return Err(DeFiError::UnsupportedPair);
        }
        
        // Créer un nouveau pool
        let pool_id = Uuid::new_v4();
        let pool = LiquidityPool::new(pool_id, pair.clone(), fee_percent);
        self.liquidity_pools.insert(pair, pool);
        
        Ok(pool_id)
    }
    
    // Ajouter de la liquidité à un pool
    pub fn add_liquidity(
        &mut self, 
        token_a: TokenId, 
        token_b: TokenId,
        amount_a: Decimal, 
        amount_b: Decimal,
        provider: Address
    ) -> Result<Decimal, DeFiError> {
        // Trouver le pool
        let pair = self.get_or_create_pair(token_a, token_b)?;
        let pool = self.liquidity_pools.get_mut(&pair).ok_or(DeFiError::UnsupportedPair)?;
        
        // Ajouter la liquidité
        let lp_tokens = pool.add_liquidity(provider, amount_a, amount_b)?;
        
        Ok(lp_tokens)
    }
    
    // Retirer de la liquidité d'un pool
    pub fn remove_liquidity(
        &mut self,
        token_a: TokenId,
        token_b: TokenId,
        lp_amount: Decimal,
        provider: Address
    ) -> Result<(Decimal, Decimal), DeFiError> {
        // Trouver le pool
        let pair = self.get_pair(token_a, token_b)?;
        let pool = self.liquidity_pools.get_mut(&pair).ok_or(DeFiError::UnsupportedPair)?;
        
        // Retirer la liquidité
        let (amount_a, amount_b) = pool.remove_liquidity(&provider, lp_amount)?;
        
        Ok((amount_a, amount_b))
    }
    
    // Exécuter un swap
    pub fn swap(
        &mut self,
        token_in: TokenId,
        token_out: TokenId,
        amount_in: Decimal
    ) -> Result<Decimal, DeFiError> {
        // Trouver le pool
        let pair = self.get_pair(token_in, token_out)?;
        let pool = self.liquidity_pools.get_mut(&pair).ok_or(DeFiError::UnsupportedPair)?;
        
        // Exécuter le swap
        let amount_out = pool.swap(token_in, amount_in)?;
        
        Ok(amount_out)
    }
    
    // ====== GESTION DES PRÊTS ======
    
    // Créer une nouvelle plateforme de prêt
    pub fn create_lending_platform(&mut self, oracle_id: Uuid) -> Result<Uuid, DeFiError> {
        // Vérifier que l'oracle existe
        let oracle = self.oracle_providers.get(&oracle_id)
            .ok_or_else(|| DeFiError::OracleError("Oracle inconnu".to_string()))?
            .clone();
        
        // Créer la plateforme
        let platform_id = Uuid::new_v4();
        let platform = LendingPlatform {
            id: platform_id,
            pools: HashMap::new(),
            collateral_ratios: HashMap::new(),
            liquidation_threshold: Decimal::new(150, 2), // 150%
            oracle,
        };
        
        self.lending_platforms.insert(platform_id, platform);
        
        Ok(platform_id)
    }
    
    // Ajouter un pool de prêt pour un token
    pub fn add_lending_pool(
        &mut self,
        platform_id: Uuid,
        token: TokenId,
        collateral_ratio: Decimal
    ) -> Result<(), DeFiError> {
        let platform = self.lending_platforms.get_mut(&platform_id)
            .ok_or_else(|| DeFiError::TransactionError("Plateforme de prêt inconnue".to_string()))?;
        
        // Vérifier que le ratio est raisonnable
        if collateral_ratio < Decimal::new(110, 2) {
            return Err(DeFiError::InsufficientCollateral);
        }
        
        // Créer le pool
        let lending_pool = LendingPool {
            token,
            total_supplied: Decimal::ZERO,
            total_borrowed: Decimal::ZERO,
            utilization_rate: Decimal::ZERO,
            supply_apy: Decimal::new(2, 2), // 2% APY initial
            borrow_apy: Decimal::new(5, 2), // 5% APY initial
            suppliers: HashMap::new(),
            borrowers: HashMap::new(),
        };
        
        platform.pools.insert(token, lending_pool);
        platform.collateral_ratios.insert(token, collateral_ratio);
        
        Ok(())
    }
    
    // Fournir des actifs à un pool de prêt
    pub fn supply_to_lending_pool(
        &mut self,
        platform_id: Uuid,
        token: TokenId,
        amount: Decimal,
        supplier: Address
    ) -> Result<(), DeFiError> {
        let platform = self.lending_platforms.get_mut(&platform_id)
            .ok_or_else(|| DeFiError::TransactionError("Plateforme de prêt inconnue".to_string()))?;
        
        let pool = platform.pools.get_mut(&token)
            .ok_or_else(|| DeFiError::TransactionError("Pool de prêt inconnu".to_string()))?;
        
        // Créer ou mettre à jour la position de fourniture
        let supply = pool.suppliers.entry(supplier).or_insert(Supply {
            address: supplier,
            amount: Decimal::ZERO,
            timestamp: Utc::now(),
        });
        
        supply.amount += amount;
        supply.timestamp = Utc::now();
        
        // Mettre à jour les totaux du pool
        pool.total_supplied += amount;
        
        // Recalculer le taux d'utilisation
        if pool.total_supplied > Decimal::ZERO {
            pool.utilization_rate = pool.total_borrowed / pool.total_supplied;
        }
        
        // Recalculer les APY
        recalculate_lending_rates(pool);
        
        Ok(())
    }
    
    // Retirer des actifs d'un pool de prêt
    pub fn withdraw_from_lending_pool(
        &mut self,
        platform_id: Uuid,
        token: TokenId,
        amount: Decimal,
        supplier: Address
    ) -> Result<(), DeFiError> {
        let platform = self.lending_platforms.get_mut(&platform_id)
            .ok_or_else(|| DeFiError::TransactionError("Plateforme de prêt inconnue".to_string()))?;
        
        let pool = platform.pools.get_mut(&token)
            .ok_or_else(|| DeFiError::TransactionError("Pool de prêt inconnu".to_string()))?;
        
        // Vérifier que le fournisseur a suffisamment d'actifs
        let supply = pool.suppliers.get_mut(&supplier)
            .ok_or_else(|| DeFiError::InsufficientTokens)?;
            
        if supply.amount < amount {
            return Err(DeFiError::InsufficientTokens);
        }
        
        // Vérifier que le pool a assez de liquidité
        let available_liquidity = pool.total_supplied - pool.total_borrowed;
        if amount > available_liquidity {
            return Err(DeFiError::InsufficientLiquidity);
        }
        
        // Mettre à jour la position
        supply.amount -= amount;
        supply.timestamp = Utc::now();
        
        // Retirer complètement si le solde est nul
        if supply.amount == Decimal::ZERO {
            pool.suppliers.remove(&supplier);
        }
        
        // Mettre à jour les totaux du pool
        pool.total_supplied -= amount;
        
        // Recalculer le taux d'utilisation
        if pool.total_supplied > Decimal::ZERO {
            pool.utilization_rate = pool.total_borrowed / pool.total_supplied;
        } else {
            pool.utilization_rate = Decimal::ZERO;
        }
        
        // Recalculer les APY
        recalculate_lending_rates(pool);
        
        Ok(())
    }
    
    // ====== GESTION DES ORACLES ======
    
    // Créer un nouveau fournisseur d'oracle
    pub fn create_oracle_provider(&mut self, update_interval: chrono::Duration) -> Uuid {
        let oracle_id = Uuid::new_v4();
        let oracle = OracleProvider::new(update_interval);
        
        self.oracle_providers.insert(oracle_id, Arc::new(oracle));
        
        oracle_id
    }
    
    // Ajouter une source de prix à l'oracle
    pub fn add_price_source
