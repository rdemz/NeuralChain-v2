use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use std::path::Path;
use tempfile::tempdir;
use neuralchain::storage::OptimizedStorage;
use neuralchain::block::Block;
use neuralchain::transaction::Transaction;
use neuralchain::transaction::{TransactionType, SignatureScheme};

fn storage_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("StorageOperations");
    group.measurement_time(Duration::from_secs(10));
    
    // Créer un répertoire temporaire pour les tests
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_db");
    
    // Initialiser le stockage
    let storage = OptimizedStorage::new(db_path.to_str().unwrap()).unwrap();
    
    // Préparer des données de test
    let block = Block::genesis();
    
    // Benchmark pour l'écriture de blocs
    group.bench_function("save_block", |b| {
        b.iter(|| {
            black_box(storage.save_block(&block).unwrap())
        })
    });
    
    // Benchmark pour la lecture de blocs
    group.bench_function("get_block", |b| {
        // S'assurer que le bloc est sauvegardé
        storage.save_block(&block).unwrap();
        
        b.iter(|| {
            black_box(storage.get_block(&block.hash).unwrap())
        })
    });
    
    // Benchmark pour la lecture de blocs par hauteur
    group.bench_function("get_block_by_height", |b| {
        b.iter(|| {
            black_box(storage.get_block_by_height(0).unwrap())
        })
    });
    
    // Benchmark pour la gestion des soldes de compte
    let test_account = [1u8; 32];
    group.bench_function("update_account_balance", |b| {
        b.iter(|| {
            black_box(storage.update_account_balance(&test_account, 1000).unwrap())
        })
    });
    
    group.bench_function("get_account_balance", |b| {
        // S'assurer que le solde est initialisé
        storage.update_account_balance(&test_account, 1000).unwrap();
        
        b.iter(|| {
            black_box(storage.get_account_balance(&test_account).unwrap())
        })
    });
    
    // Benchmark pour les opérations de vidange
    group.bench_function("flush", |b| {
        b.iter(|| {
            black_box(storage.flush().unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(benches, storage_benchmark);
criterion_main!(benches);
