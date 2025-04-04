use sha2::{Sha256, Digest};
use ed25519_dalek::{Keypair, Signer, Verifier, Signature};
use anyhow::{Result, Context};

/// Calcule un hash SHA-256 des données
pub fn hash(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Crée une signature Ed25519 des données avec la clé privée fournie
pub fn create_signature(keypair: &Keypair, data: &[u8]) -> Result<Vec<u8>> {
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
    
    let public_key = VerifyingKey::from_bytes(public_key_bytes.try_into()
        .context("Clé publique invalide")?)
        .context("Impossible de créer la clé de vérification")?;
    
    let signature = Signature::from_bytes(signature_bytes.try_into()
        .context("Signature invalide")?);
    
    match public_key.verify(data, &signature) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Génère une paire de clés Ed25519
pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    use rand::rngs::OsRng;
    
    let mut csprng = OsRng;
    let keypair = Keypair::generate(&mut csprng);
    
    let private_key = keypair.to_bytes().to_vec();
    let public_key = keypair.public.to_bytes().to_vec();
    
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
    use rand::rngs::OsRng;
    
    #[test]
    fn test_hash() {
        let data = b"test data";
        let hash_result = hash(data);
        assert_eq!(hash_result.len(), 32); // SHA-256 produit des hashes de 32 octets
    }
    
    #[test]
    fn test_signature_verification() {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        let data = b"test message";
        
        let signature_result = create_signature(&keypair, data).unwrap();
        let verified = verify_signature(
            keypair.public.as_bytes(),
            data,
            &signature_result
        ).unwrap();
        
        assert!(verified);
    }
    
    #[test]
    fn test_keypair_generation() {
        let (private_key, public_key) = generate_keypair().unwrap();
        
        assert_eq!(private_key.len(), 64); // Ed25519 clé privée: 64 octets
        assert_eq!(public_key.len(), 32);  // Ed25519 clé publique: 32 octets
    }
    
    #[test]
    fn test_address_derivation() {
        let data = vec![1, 2, 3, 4, 5];
        let address = derive_address(&data);
        
        assert_eq!(address.len(), 20);
    }
}
