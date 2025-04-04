use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use std::sync::Arc;
use tokio::sync::Mutex;
use neuralchain::neural_pow::NeuralPoW;
use neuralchain::blockchain::Blockchain;
use neuralchain::block::Block;

fn neural_pow_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("NeuralPoWOperations");
    group.measurement_time(Duration::from_secs(10));
    
    // Setup - utilisons le runtime Tokio pour les tests
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        let blockchain = Arc::new(Mutex::new(Blockchain::new_empty()));
        let pow_system = NeuralPoW::new(blockchain.clone());
        
        // Créer un bloc de test
        let test_block = Block::genesis();
        let test_hash = test_block.hash;
        let test_difficulty = 1;
        
        // Benchmark pour le calcul de hash PoW
        group.bench_function("compute_pow_hash", |b| {
            b.iter(|| {
                rt.block_on(async {
                    black_box(pow_system.compute_pow_hash(&test_hash, 12345).await)
                })
            })
        });
        
        // Benchmark pour la vérification du PoW
        group.bench_function("verify_pow", |b| {
            b.iter(|| {
                rt.block_on(async {
                    black_box(pow_system.verify_pow(&test_hash, 12345, test_difficulty).await)
                })
            })
        });
        
        // Benchmark pour le mining
        group.bench_function("mine_block_limited", |b| {
            b.iter(|| {
                rt.block_on(async {
                    // Limiter à 1000 itérations pour le benchmark
                    black_box(pow_system.mine_block_limited(&test_block, 1000).await)
                })
            })
        });
        
        // Benchmark pour l'ajustement des poids neuronaux
        group.bench_function("adjust_network_weights", |b| {
            b.iter(|| {
                rt.block_on(async {
                    black_box(pow_system.adjust_network_weights().await)
                })
            })
        });
        
        // Benchmark pour l'analyse des modèles de réseau
        group.bench_function("analyze_network_patterns", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let blockchain_guard = blockchain.lock().await;
                    black_box(pow_system.analyze_network_patterns(&*blockchain_guard).await)
                })
            })
        });
    });
    
    group.finish();
}

criterion_group!(benches, neural_pow_benchmark);
criterion_main!(benches);
