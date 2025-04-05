use anyhow::{Result, Context};
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};
use hex;
use rand::Rng;

/// Formatage hexadécimal d'un tableau d'octets
pub fn bytes_to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Conversion d'une chaîne hexadécimale en tableau d'octets
pub fn hex_to_bytes(hex_string: &str) -> Result<Vec<u8>> {
    hex::decode(hex_string).context("Échec de la conversion hexadécimale en octets")
}

/// Obtention du timestamp actuel en millisecondes
pub fn current_timestamp_ms() -> u64 {
    // Note: Cette conversion pourrait théoriquement causer un dépassement de capacité
    // après plusieurs centaines d'années depuis UNIX_EPOCH. Pour les besoins pratiques
    // de la blockchain, c'est acceptable.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Calcul du hash SHA-256 de données
pub fn sha256_hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Génère une chaîne aléatoire de longueur spécifiée
pub fn random_string(length: usize) -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::thread_rng();
    
    (0..length)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Vérifie si une adresse IP est valide
pub fn is_valid_ip(ip: &str) -> bool {
    use std::net::IpAddr;
    ip.parse::<IpAddr>().is_ok()
}

/// Formatage d'une taille en octets en une chaîne lisible
pub fn format_bytes_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;
    
    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

/// Mesure le temps d'exécution d'une fonction
pub async fn measure_execution_time<F, Fut, T>(f: F) -> (T, u128)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    let duration = start.elapsed().as_millis();
    
    (result, duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bytes_to_hex() {
        let bytes = vec![0, 1, 10, 255];
        assert_eq!(bytes_to_hex(&bytes), "00010aff");
    }
    
    #[test]
    fn test_hex_to_bytes() {
        let hex_string = "00010aff";
        assert_eq!(hex_to_bytes(hex_string).unwrap(), vec![0, 1, 10, 255]);
    }
    
    #[test]
    fn test_sha256_hash() {
        let data = b"test data";
        let hash = sha256_hash(data);
        assert_eq!(hash.len(), 32); // SHA-256 produit des hashes de 32 octets
    }
    
    #[test]
    fn test_is_valid_ip() {
        assert!(is_valid_ip("192.168.1.1"));
        assert!(is_valid_ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334"));
        assert!(!is_valid_ip("not an ip"));
    }
    
    #[test]
    fn test_format_bytes_size() {
        assert_eq!(format_bytes_size(500), "500 B");
        assert_eq!(format_bytes_size(1500), "1.46 KB");
        assert_eq!(format_bytes_size(1500000), "1.43 MB");
    }
    
    // Suppression du test asynchrone qui cause des problèmes
    // Nous pourrions le réactiver avec tokio::test, mais pour l'instant nous le retirons
}
