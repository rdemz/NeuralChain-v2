use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use neuralchain::consensus::ConsensusEngine;
use neuralchain::blockchain::Blockchain;
use neuralchain::neural_pow::NeuralPoW;
use neuralchain::block::Block;
use std::sync::Arc;
use tokio::sync::Mutex;

fn consensus_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ConsensusOperations");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(20);
    
    // Setup - utilisons le runtime Tokio pour les tests
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    rt.block_on(async {
        let blockchain = Arc::new(Mutex::new(Blockchain::new_empty()));
        let pow_system = NeuralPoW::new(blockchain.clone());
        let consensus_engine = ConsensusEngine::new(
            blockchain.clone(),
            Arc::new(pow_system),
            Default::default(),
        );
        
        // Benchmark block verification
        group.bench_function("verify_block", |b| {
            b.iter(|| {
                let block = rt.block_on(async {
                    let test_block = Block::genesis(); // Utilisons le bloc de genèse pour simplifier
                    black_box(consensus_engine.verify_block(&test_block).await)
                });
                black_box(block)
            })
        });
        
        // Benchmark PoW difficulty calculation
        group.bench_function("calculate_difficulty", |b| {
            b.iter(|| {
                let difficulty = rt.block_on(async {
                    let blockchain_guard = blockchain.lock().await;
                    let last_block = blockchain_guard.get_block_by_height(0).unwrap_or(Block::genesis());
                    black_box(consensus_engine.calculate_next_difficulty(&last_block).await)
                });
                black_box(difficulty)
            })
        });
        
        // Benchmark neural weights adjustment
        group.bench_function("neural_weights_adjustment", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let pow_system = consensus_engine.get_pow_system();
                    black_box(pow_system.adjust_network_weights().await)
                })
            })
        });
        
        // Benchmark pour la création de bloc
        group.bench_function("create_block", |b| {
            b.iter(|| {
                rt.block_on(async {
                    let blockchain_guard = blockchain.lock().await;
                    let parent = blockchain_guard.get_latest_block().unwrap_or(Block::genesis());
                    let parent_hash = parent.hash;
                    let height = parent.height + 1;
                    let timestamp = chrono::Utc::now().timestamp() as u64;
                    drop(blockchain_guard);
                    
                    black_box(consensus_engine.create_new_block(
                        parent_hash,
                        height,
                        timestamp,
                        vec![], // transactions vides pour le benchmark
                        1234567890, // nonce de test
                    ).await)
                })
            })
        });
    });
    
    group.finish();
}

criterion_group!(benches, consensus_benchmark);
criterion_main!(benches);
