use sha2::{Sha256, Digest};
use ed25519_dalek::{SigningKey, Signature, Verifier, Signer};
use anyhow::Result;
use rand::rngs::OsRng;

/// Calcule un hash SHA-256 des données
pub fn hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Crée une signature Ed25519 des données avec la clé privée fournie
pub fn create_signature(keypair: &SigningKey, data: &[u8]) -> Result<Vec<u8>> {
    let signature = keypair.sign(data);
    Ok(signature.to_bytes().to_vec())
}

/// Vérifie une signature Ed25519 avec la clé publique correspondante
pub fn verify_signature(
    public_key_bytes: &[u8],
    data: &[u8],
    signature_bytes: &[u8]
) -> Result<bool> {
    use ed25519_dalek::VerifyingKey;
    
    // Convertir la clé publique
    if public_key_bytes.len() != 32 {
        return Ok(false);
    }
    
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(public_key_bytes);
    
    let public_key = match VerifyingKey::from_bytes(&key_bytes) {
        Ok(key) => key,
        Err(_) => return Ok(false),
    };
    
    // Convertir la signature
    if signature_bytes.len() != 64 {
        return Ok(false);
    }
    
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(signature_bytes);
    
    let signature = Signature::from_bytes(&sig_bytes);
    
    // Vérifier la signature
    match public_key.verify(data, &signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Génère une paire de clés Ed25519
pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    let mut csprng = OsRng;
    let keypair = SigningKey::generate(&mut csprng);
    
    let private_key = keypair.to_bytes().to_vec();
    let public_key = keypair.verifying_key().to_bytes().to_vec();
    
    Ok((private_key, public_key))
}

/// Hache des données deux fois avec SHA-256 (comme dans Bitcoin)
pub fn double_sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher1 = Sha256::new();
    hasher1.update(data);
    let first_hash = hasher1.finalize();
    
    let mut hasher2 = Sha256::new();
    hasher2.update(&first_hash);
    hasher2.finalize().to_vec()
}

/// Dérive une adresse à partir d'une clé publique
pub fn derive_address(public_key: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(public_key);
    
    // Prendre les 20 premiers octets (comme Ethereum)
    hasher.finalize()[0..20].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hash() {
        let data = b"test data";
        let hash_result = hash(data);
        assert_eq!(hash_result.len(), 32); // SHA-256 produit des hashes de 32 octets
    }
    
    #[test]
    fn test_signature_verification() {
        let mut csprng = OsRng;
        let keypair = SigningKey::generate(&mut csprng);
        let data = b"test message";
        
        let signature_result = create_signature(&keypair, data).unwrap();
        let verified = verify_signature(
            &keypair.verifying_key().to_bytes(),
            data,
            &signature_result
        ).unwrap();
        
        assert!(verified);
    }
    
    #[test]
    fn test_keypair_generation() {
        let (private_key, public_key) = generate_keypair().unwrap();
        
        assert_eq!(private_key.len(), 32); // Ed25519 clé privée: 32 octets
        assert_eq!(public_key.len(), 32);  // Ed25519 clé publique: 32 octets
    }
    
    #[test]
    fn test_address_derivation() {
        let data = vec![1, 2, 3, 4, 5];
        let address = derive_address(&data);
        
        assert_eq!(address.len(), 20);
    }
}
