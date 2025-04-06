//! Utilitaires système cross-platform pour NeuralChain

use std::time::{Instant, Duration};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};
use std::sync::Arc;
use parking_lot::RwLock;

/// Monitor système cross-platform
pub struct SystemMonitor {
    sys: RwLock<System>,
    last_refresh: RwLock<Instant>,
    refresh_interval: Duration,
}

impl SystemMonitor {
    /// Crée un nouveau moniteur système
    pub fn new() -> Self {
        let sys = System::new_all();
        Self {
            sys: RwLock::new(sys),
            last_refresh: RwLock::new(Instant::now()),
            refresh_interval: Duration::from_millis(500),
        }
    }
    
    /// Actualise les données du système si nécessaire
    fn refresh_if_needed(&self) {
        let now = Instant::now();
        let should_refresh = {
            let last = *self.last_refresh.read();
            now.duration_since(last) >= self.refresh_interval
        };
        
        if should_refresh {
            let mut sys = self.sys.write();
            sys.refresh_all();
            *self.last_refresh.write() = now;
        }
    }
    
    /// Renvoie l'utilisation de la mémoire en pourcentage
    pub fn get_memory_usage(&self) -> f64 {
        self.refresh_if_needed();
        let sys = self.sys.read();
        let used = sys.used_memory() as f64;
        let total = sys.total_memory() as f64;
        
        if total > 0.0 {
            (used / total) * 100.0
        } else {
            0.0
        }
    }
    
    /// Renvoie la charge CPU globale en pourcentage
    pub fn get_cpu_usage(&self) -> f64 {
        self.refresh_if_needed();
        let sys = self.sys.read();
        
        let mut total = 0.0;
        let cpu_count = sys.cpus().len();
        
        if cpu_count == 0 {
            return 0.0;
        }
        
        for cpu in sys.cpus() {
            total += cpu.cpu_usage();
        }
        
        total / cpu_count as f64
    }
    
    /// Renvoie la charge du processus courant en pourcentage
    pub fn get_process_usage(&self) -> f64 {
        self.refresh_if_needed();
        let sys = self.sys.read();
        
        match sys.process(sysinfo::get_current_pid().ok()?) {
            Some(process) => process.cpu_usage(),
            None => 0.0,
        }
    }
}

/// Optimiseur de performance cross-platform
pub struct PerformanceOptimizer;

impl PerformanceOptimizer {
    /// Tente d'optimiser la priorité du thread actuel
    pub fn optimize_thread_priority() -> Result<(), String> {
        #[cfg(target_os = "windows")]
        {
            // Optimisation spécifique à Windows
            unsafe {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                use windows_sys::Win32::System::Threading::THREAD_PRIORITY_ABOVE_NORMAL;
                
                let handle = GetCurrentThread();
                if SetThreadPriority(handle, THREAD_PRIORITY_ABOVE_NORMAL) == 0 {
                    return Err("Échec de l'optimisation du thread".to_string());
                }
            }
        }
        
        // Pour les autres systèmes d'exploitation, on ne fait rien de spécial
        Ok(())
    }
    
    /// Mesure la durée d'exécution d'une fonction
    pub fn measure_execution<F, T>(func: F) -> (T, Duration)
    where
        F: FnOnce() -> T
    {
        let start = Instant::now();
        let result = func();
        let elapsed = start.elapsed();
        
        (result, elapsed)
    }
    
    /// Obtient un timestamp haute précision
    pub fn high_precision_timestamp() -> u64 {
        let now = Instant::now();
        // Convertir en nanosecondes depuis le début du programme
        now.elapsed().as_nanos() as u64
    }
}

/// Gestionnaire de priorité de processus
pub struct ProcessPriorityManager;

impl ProcessPriorityManager {
    /// Tente d'augmenter la priorité du processus actuel
    pub fn increase_process_priority() -> Result<(), String> {
        #[cfg(target_os = "windows")]
        {
            unsafe {
                use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass};
                use windows_sys::Win32::System::Threading::HIGH_PRIORITY_CLASS;
                
                let handle = GetCurrentProcess();
                if SetPriorityClass(handle, HIGH_PRIORITY_CLASS) == 0 {
                    return Err("Échec de l'augmentation de la priorité du processus".to_string());
                }
            }
        }
        
        Ok(())
    }
}

/// Module de synchronisation haute précision
pub mod high_precision {
    use std::time::{Instant, Duration};
    
    /// Obtient un compteur de performance haute précision
    pub fn get_performance_counter() -> u64 {
        #[cfg(target_os = "windows")]
        {
            // Implémentation Windows spécifique si nécessaire
            unsafe {
                use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, LARGE_INTEGER};
                let mut counter: LARGE_INTEGER = 0;
                QueryPerformanceCounter(&mut counter);
                counter as u64
            }
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // Implémentation portable
            Instant::now().elapsed().as_nanos() as u64
        }
    }
    
    /// Obtient la fréquence du compteur de performance
    pub fn get_performance_frequency() -> u64 {
        #[cfg(target_os = "windows")]
        {
            // Implémentation Windows spécifique si nécessaire
            unsafe {
                use windows_sys::Win32::System::Performance::{QueryPerformanceFrequency, LARGE_INTEGER};
                let mut frequency: LARGE_INTEGER = 0;
                QueryPerformanceFrequency(&mut frequency);
                frequency as u64
            }
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // Fréquence simulée: nanoseconde = 10^9 Hz
            1_000_000_000
        }
    }
    
    /// Effectue une attente de haute précision
    pub fn precise_sleep(duration: Duration) {
        let target = Instant::now() + duration;
        while Instant::now() < target {
            // Attente active pour une précision maximale
            std::hint::spin_loop();
        }
    }
}

pub mod graphics {
    pub struct GraphicsAccelerator;
    
    impl GraphicsAccelerator {
        pub fn new() -> Result<Self, String> {
            #[cfg(target_os = "windows")]
            {
                // Utiliser une autre approche sans dépendre de windows-sys::Graphics
                // Par exemple, utiliser directement d'autres crates comme wgpu
            }
            
            #[cfg(not(target_os = "windows"))]
            {
                // Implémentation pour autres plateformes
            }
            
            Ok(Self {})
        }
        
        pub fn accelerate_computation(&self) -> Result<f64, String> {
            // Implémentation d'accélération
            Ok(1.0) // Facteur d'accélération
        }
    }
}
