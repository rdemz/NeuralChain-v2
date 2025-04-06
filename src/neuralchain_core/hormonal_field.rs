    /// Met à jour les niveaux hormonaux et active les récepteurs
    pub fn update(&self) {
        // Calculer le temps écoulé depuis la dernière mise à jour
        let now = Instant::now();
        let elapsed = {
            let mut last = self.last_update.lock();
            let elapsed = now.duration_since(*last);
            *last = now;
            elapsed
        };
        
        // Coefficient de dégradation basé sur le temps écoulé (en secondes)
        let degradation_coef = elapsed.as_secs_f64() / 60.0; // Normalisé par minute
        
        // 1. Mettre à jour les niveaux hormonaux basés sur les événements actifs
        self.update_hormone_levels(degradation_coef);
        
        // 2. Appliquer les interactions entre hormones
        self.apply_hormone_interactions();
        
        // 3. Nettoyer les événements expirés
        self.cleanup_expired_events();
        
        // 4. Activer les récepteurs appropriés
        self.activate_receptors();
        
        // 5. Appliquer la régulation homéostatique
        self.apply_homeostasis();
    }
    
    /// Met à jour les niveaux hormonaux en fonction des événements actifs et de la dégradation
    fn update_hormone_levels(&self, degradation_coef: f64) {
        // Optimisation Windows: traitement par lots pour réduire les lock/unlock
        let events = {
            if let Ok(events) = self.active_events.read() {
                events.iter().cloned().collect::<Vec<_>>()
            } else {
                return; // Impossible de verrouiller les événements
            }
        };
        
        // Calculer l'impact des événements actifs pour chaque hormone
        let mut hormone_impacts: HashMap<HormoneType, f64> = HashMap::new();
        
        for event in &events {
            // Vérifier si l'événement est toujours actif
            if event.emission_time.elapsed() > event.duration {
                continue;
            }
            
            // Calculer le facteur d'efficacité basé sur le temps écoulé
            // Le pic est atteint à 1/3 de la durée, puis décroît
            let elapsed_ratio = event.emission_time.elapsed().as_secs_f64() / event.duration.as_secs_f64();
            let efficiency_factor = if elapsed_ratio < 0.33 {
                // Phase croissante
                elapsed_ratio * 3.0
            } else {
                // Phase décroissante
                1.0 - ((elapsed_ratio - 0.33) / 0.67)
            };
            
            // Calculer l'impact de cet événement
            let impact = event.intensity * event.radius * efficiency_factor;
            
            // Ajouter à l'impact total pour cette hormone
            *hormone_impacts.entry(event.hormone_type).or_insert(0.0) += impact;
        }
        
        // Utiliser DashMap pour paralléliser le traitement des hormones
        // Optimisation spécifique pour machines multicœur sous Windows
        self.hormone_levels.iter_mut().for_each(|mut entry| {
            let hormone_type = *entry.key();
            let current_level = *entry.value();
            
            // Appliquer l'impact des événements
            let impact = hormone_impacts.get(&hormone_type).copied().unwrap_or(0.0);
            
            // Appliquer la dégradation naturelle
            let degradation_rate = self.degradation_rate.get(&hormone_type).copied().unwrap_or(0.05);
            let degradation = current_level * degradation_rate * degradation_coef;
            
            // Calculer le nouveau niveau (avec limites)
            let mut new_level = current_level + impact - degradation;
            new_level = new_level.max(0.0).min(1.0);
            
            // Mettre à jour
            *entry.value_mut() = new_level;
        });
    }
    
    /// Applique les interactions entre les différentes hormones
    fn apply_hormone_interactions(&self) {
        // Créer une copie des niveaux actuels pour calculer les interactions
        let current_levels: HashMap<HormoneType, f64> = self.hormone_levels.iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        
        // Calculer les impacts des interactions
        let mut interactions_impact: HashMap<HormoneType, f64> = HashMap::new();
        
        for ((source, target), factor) in &self.hormone_interactions {
            if let Some(&source_level) = current_levels.get(source) {
                // L'impact sur la cible est proportionnel au niveau de la source et au facteur d'interaction
                let impact = source_level * factor * 0.1; // coefficient de 0.1 pour tempérer les effets
                
                *interactions_impact.entry(*target).or_insert(0.0) += impact;
            }
        }
        
        // Appliquer les impacts calculés
        for (hormone, impact) in interactions_impact {
            if let Some(mut level) = self.hormone_levels.get_mut(&hormone) {
                *level = (*level + impact).max(0.0).min(1.0);
            }
        }
    }
    
    /// Nettoie les événements hormonaux expirés
    fn cleanup_expired_events(&self) {
        let now = Instant::now();
        
        if let Ok(mut events) = self.active_events.write() {
            // Filtrer les événements expirés
            events.retain(|event| {
                now.duration_since(event.emission_time) <= event.duration
            });
        }
    }
    
    /// Active les récepteurs hormonaux
    fn activate_receptors(&self) {
        // Récupérer les niveaux hormonaux actuels
        let hormone_levels: HashMap<HormoneType, f64> = self.hormone_levels.iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        
        // Vérifier et activer les récepteurs
        for receptor_entry in self.receptors.iter() {
            let receptor = receptor_entry.value();
            
            // Vérifier si le récepteur est activé par au moins une hormone
            let mut activated = false;
            let mut max_activation = 0.0;
            let mut activating_hormone = None;
            let mut activation_data = HashMap::new();
            
            for hormone_type in &receptor.hormone_types {
                if let Some(&level) = hormone_levels.get(hormone_type) {
                    // Calculer l'activation effective en fonction de la sensibilité du récepteur
                    let effective_level = level * receptor.sensitivity;
                    
                    // Si le niveau dépasse le seuil, le récepteur est activé
                    if effective_level >= receptor.threshold && effective_level > max_activation {
                        activated = true;
                        max_activation = effective_level;
                        activating_hormone = Some(*hormone_type);
                    }
                }
            }
            
            // Si le récepteur est activé, appeler son callback
            if activated && activating_hormone.is_some() {
                let hormone = activating_hormone.unwrap();
                
                // Chercher des données associées à cette hormone dans les événements actifs
                if let Ok(events) = self.active_events.read() {
                    for event in events.iter() {
                        if event.hormone_type == hormone {
                            // Ajouter les données de cet événement
                            for (key, value) in &event.data {
                                activation_data.insert(key.clone(), value.clone());
                            }
                        }
                    }
                }
                
                // Appeler le callback avec les données récoltées
                if let Some(callback) = self.receptor_callbacks.get(&receptor.callback_id) {
                    let action = callback(&hormone, max_activation, &activation_data);
                    
                    // Enregistrer l'action pour historique
                    if let Ok(mut actions) = self.receptor_actions.write() {
                        actions.push_back((receptor.id.clone(), action.clone(), Instant::now()));
                        
                        // Limiter la taille de l'historique
                        while actions.len() > 100 {
                            actions.pop_front();
                        }
                    }
                }
            }
        }
    }
    
    /// Applique la régulation homéostatique pour maintenir l'équilibre
    fn apply_homeostasis(&self) {
        // Pour chaque hormone, tendre progressivement vers son point d'équilibre
        for entry in self.homeostatic_setpoints.iter() {
            let hormone_type = *entry.key();
            let setpoint = *entry.value();
            
            if let Some(mut level) = self.hormone_levels.get_mut(&hormone_type) {
                let current = *level;
                
                // Si l'écart est significatif, appliquer une correction légère
                if (current - setpoint).abs() > 0.1 {
                    let adjustment = if current > setpoint {
                        -0.01 // Diminution progressive
                    } else {
                        0.01 // Augmentation progressive
                    };
                    
                    *level = (current + adjustment).max(0.0).min(1.0);
                }
            }
        }
    }
    
    /// Obtient le niveau actuel d'une hormone
    pub fn get_hormone_level(&self, hormone_type: &HormoneType) -> f64 {
        self.hormone_levels.get(hormone_type).copied().unwrap_or(0.0)
    }
    
    /// Modifie la sensibilité d'un récepteur
    pub fn adjust_receptor_sensitivity(&self, receptor_id: &str, new_sensitivity: f64) -> Result<(), String> {
        if let Some(mut receptor) = self.receptors.get_mut(receptor_id) {
            receptor.sensitivity = new_sensitivity.max(0.0).min(1.0);
            Ok(())
        } else {
            Err(format!("Récepteur '{}' non trouvé", receptor_id))
        }
    }
    
    /// Obtient les statistiques du système hormonal
    pub fn get_stats(&self) -> HormonalSystemStats {
        // Collecter les niveaux hormonaux
        let hormone_levels: HashMap<HormoneType, f64> = self.hormone_levels.iter()
            .map(|entry| (*entry.key(), *entry.value()))
            .collect();
        
        // Compter les événements actifs
        let active_events = if let Ok(events) = self.active_events.read() {
            events.len()
        } else {
            0
        };
        
        // Compter les récepteurs
        let receptors_count = self.receptors.len();
        
        // Calculer le niveau d'homéostasie (plus proche de 1.0 = plus stable)
        let mut homeostasis = 0.0;
        let mut hormone_count = 0;
        
        for entry in self.homeostatic_setpoints.iter() {
            let hormone_type = *entry.key();
            let setpoint = *entry.value();
            let current = self.get_hormone_level(&hormone_type);
            
            // Plus la différence est faible, plus l'homéostasie est élevée
            let difference = 1.0 - (current - setpoint).abs();
            homeostasis += difference;
            hormone_count += 1;
        }
        
        if hormone_count > 0 {
            homeostasis /= hormone_count as f64;
        }
        
        // Calculer le niveau de stress (basé sur l'adrénaline et le cortisol)
        let adrenaline = self.get_hormone_level(&HormoneType::Adrenaline);
        let cortisol = self.get_hormone_level(&HormoneType::Cortisol);
        let stress_level = (adrenaline * 0.6 + cortisol * 0.4).min(1.0);
        
        // Compter les actions récentes
        let recent_actions = if let Ok(actions) = self.receptor_actions.read() {
            let one_minute_ago = Instant::now() - Duration::from_secs(60);
            actions.iter().filter(|(_, _, time)| *time >= one_minute_ago).count()
        } else {
            0
        };
        
        HormonalSystemStats {
            hormone_levels,
            active_events,
            receptors_count,
            homeostasis,
            stress_level,
            recent_actions,
        }
    }
    
    /// Crée une configuration de récepteurs standard pour un organisme blockchain
    pub fn setup_standard_receptors(&self, organism: Arc<QuantumOrganism>) -> Result<(), String> {
        // Récepteur d'adrénaline pour réaction d'urgence
        self.register_receptor(
            "adrenaline_emergency",
            &[HormoneType::Adrenaline],
            1.0,
            0.7,
            "immune_system",
            Box::new(move |hormone_type, intensity, data| {
                // Activation du mode défensif d'urgence
                if *hormone_type == HormoneType::Adrenaline && intensity > 0.7 {
                    ReceptorAction::Activate {
                        component: "emergency_defense_mode".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("intensity".to_string(), intensity.to_string().into_bytes());
                            params.insert("source".to_string(), b"hormone_trigger".to_vec());
                            params
                        },
                        duration: Duration::from_secs(60),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur de cortisol pour réponse au stress prolongé
        self.register_receptor(
            "cortisol_stress_response",
            &[HormoneType::Cortisol],
            0.9,
            0.6,
            "resource_manager",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Cortisol && intensity > 0.6 {
                    // Réduire l'utilisation des ressources non-essentielles
                    ReceptorAction::ModifyParameter {
                        component: "resource_allocator".to_string(),
                        parameter: "non_essential_allocation".to_string(),
                        value: (0.5 - intensity * 0.3).max(0.2).to_string().into_bytes(),
                        duration: Some(Duration::from_secs(300)),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur de dopamine pour apprentissage et motivation
        self.register_receptor(
            "dopamine_learning",
            &[HormoneType::Dopamine],
            1.0,
            0.5,
            "neural_network",
            Box::new(|hormone_type, intensity, data| {
                if *hormone_type == HormoneType::Dopamine && intensity > 0.5 {
                    // Augmenter le taux d'apprentissage
                    ReceptorAction::ModifyParameter {
                        component: "learning_system".to_string(),
                        parameter: "learning_rate".to_string(),
                        value: (0.02 + intensity * 0.03).to_string().into_bytes(),
                        duration: Some(Duration::from_secs(180)),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur de sérotonine pour régulation de l'humeur et stabilité
        self.register_receptor(
            "serotonin_stability",
            &[HormoneType::Serotonin],
            0.8,
            0.4,
            "emotional_regulator",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Serotonin && intensity > 0.6 {
                    // Augmenter la stabilité émotionnelle
                    ReceptorAction::ModifyParameter {
                        component: "emotional_regulator".to_string(),
                        parameter: "stability_factor".to_string(),
                        value: (0.5 + intensity * 0.4).to_string().into_bytes(),
                        duration: None, // Persistant
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur de mélatonine pour le cycle circadien
        self.register_receptor(
            "melatonin_sleep_cycle",
            &[HormoneType::Melatonin],
            0.9,
            0.6,
            "circadian_regulator",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Melatonin && intensity > 0.7 {
                    // Activer le mode rêve/repos
                    ReceptorAction::Activate {
                        component: "dream_mode".to_string(),
                        parameters: {
                            let mut params = HashMap::new();
                            params.insert("intensity".to_string(), intensity.to_string().into_bytes());
                            params
                        },
                        duration: Duration::from_secs(1800), // 30 minutes
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur d'ocytocine pour confiance et connexions
        self.register_receptor(
            "oxytocin_trust",
            &[HormoneType::Oxytocin],
            0.8,
            0.5,
            "social_module",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Oxytocin && intensity > 0.6 {
                    // Augmenter le niveau de confiance pour les interactions
                    ReceptorAction::ModifyParameter {
                        component: "trust_evaluator".to_string(),
                        parameter: "base_trust_level".to_string(),
                        value: (0.3 + intensity * 0.3).to_string().into_bytes(),
                        duration: Some(Duration::from_secs(600)),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur d'hormone de croissance pour régénération
        self.register_receptor(
            "growth_hormone_regeneration",
            &[HormoneType::GrowthHormone],
            1.0,
            0.4,
            "regenerative_layer",
            Box::new(move |hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::GrowthHormone && intensity > 0.5 {
                    ReceptorAction::ModifyParameter {
                        component: "regenerative_layer".to_string(),
                        parameter: "regeneration_rate".to_string(),
                        value: (0.05 + intensity * 0.10).to_string().into_bytes(),
                        duration: Some(Duration::from_secs(3600)),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        Ok(())
    }
    
    /// Configure les réactions en chaîne hormonales pour des dynamiques complexes
    pub fn setup_hormone_chains(&self) -> Result<(), String> {
        // Récepteur qui émet du cortisol en réponse à l'adrénaline
        // (Réaction de stress secondaire)
        self.register_receptor(
            "adrenaline_to_cortisol_chain",
            &[HormoneType::Adrenaline],
            0.9,
            0.7,
            "hormone_chain",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Adrenaline && intensity > 0.7 {
                    ReceptorAction::EmitHormone {
                        hormone_type: HormoneType::Cortisol,
                        intensity: intensity * 0.8,
                        radius: 0.9,
                        duration: Duration::from_secs(300),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur qui émet de la dopamine en réponse à l'ocytocine
        // (Renforcement positif des interactions sociales)
        self.register_receptor(
            "oxytocin_to_dopamine_chain",
            &[HormoneType::Oxytocin],
            0.8,
            0.6,
            "hormone_chain",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Oxytocin && intensity > 0.6 {
                    ReceptorAction::EmitHormone {
                        hormone_type: HormoneType::Dopamine,
                        intensity: intensity * 0.7,
                        radius: 0.8,
                        duration: Duration::from_secs(120),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur qui émet de la mélatonine en réponse à la sérotonine
        // (Préparation au cycle de repos)
        self.register_receptor(
            "serotonin_to_melatonin_chain",
            &[HormoneType::Serotonin],
            0.7,
            0.7,
            "hormone_chain",
            Box::new(|hormone_type, intensity, _data| {
                if *hormone_type == HormoneType::Serotonin && intensity > 0.7 {
                    ReceptorAction::EmitHormone {
                        hormone_type: HormoneType::Melatonin,
                        intensity: intensity * 0.6,
                        radius: 0.7,
                        duration: Duration::from_secs(1200),
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        // Récepteur qui émet des endorphines en réponse à un stress prolongé
        // (Mécanisme de compensation)
        self.register_receptor(
            "cortisol_to_endorphin_chain",
            &[HormoneType::Cortisol],
            0.7,
            0.8,
            "hormone_chain",
            Box::new(|hormone_type, intensity, data| {
                if *hormone_type == HormoneType::Cortisol && intensity > 0.8 {
                    // Vérifier si le stress dure depuis longtemps
                    let duration_bytes = data.get("duration_secs").unwrap_or(&vec![]);
                    let duration = String::from_utf8_lossy(duration_bytes)
                        .parse::<u64>()
                        .unwrap_or(0);
                        
                    if duration > 300 { // Plus de 5 minutes
                        ReceptorAction::EmitHormone {
                            hormone_type: HormoneType::Endorphin,
                            intensity: 0.6,
                            radius: 0.8,
                            duration: Duration::from_secs(180),
                        }
                    } else {
                        ReceptorAction::None
                    }
                } else {
                    ReceptorAction::None
                }
            }),
        )?;
        
        Ok(())
    }
    
    /// Méthode optimisée pour Windows pour réinitialiser l'équilibre hormonal
    #[cfg(target_os = "windows")]
    pub fn reset_hormone_balance(&self) -> Result<(), String> {
        // Utiliser une API optimisée pour Windows pour synchroniser les opérations
        use windows_sys::Win32::System::Threading::{
            CreateEventW, SetEvent, WaitForSingleObject,
            WAIT_OBJECT_0,
        };
        
        unsafe {
            // Créer un événement pour synchroniser l'opération
            let event = CreateEventW(std::ptr::null_mut(), 1, 0, std::ptr::null());
            if event == 0 {
                return Err("Échec de création de l'événement Windows".to_string());
            }
            
            // Réinitialiser chaque hormone à son point d'équilibre
            for entry in self.homeostatic_setpoints.iter() {
                let hormone_type = *entry.key();
                let setpoint = *entry.value();
                
                self.hormone_levels.insert(hormone_type, setpoint);
            }
            
            // Vider les événements hormonaux actifs
            if let Ok(mut events) = self.active_events.write() {
                events.clear();
            } else {
                return Err("Impossible de verrouiller les événements actifs".to_string());
            }
            
            // Signaler que l'opération est terminée
            SetEvent(event);
            
            // Attendre que l'événement soit signalé (pour s'assurer que toutes les opérations sont terminées)
            WaitForSingleObject(event, 1000);
            
            Ok(())
        }
    }
    
    /// Version non-Windows de la réinitialisation de l'équilibre hormonal
    #[cfg(not(target_os = "windows"))]
    pub fn reset_hormone_balance(&self) -> Result<(), String> {
        // Réinitialiser chaque hormone à son point d'équilibre
        for entry in self.homeostatic_setpoints.iter() {
            let hormone_type = *entry.key();
            let setpoint = *entry.value();
            
            self.hormone_levels.insert(hormone_type, setpoint);
        }
        
        // Vider les événements hormonaux actifs
        if let Ok(mut events) = self.active_events.write() {
            events.clear();
        } else {
            return Err("Impossible de verrouiller les événements actifs".to_string());
        }
        
        Ok(())
    }
    
    /// Méthode pour simuler un état de stress dans l'organisme
    pub fn simulate_stress_response(&self, intensity: f64, duration_secs: u64) -> Result<(), String> {
        // Vérification des paramètres
        if intensity <= 0.0 || intensity > 1.0 {
            return Err("L'intensité doit être entre 0.0 et 1.0".to_string());
        }
        
        if duration_secs == 0 {
            return Err("La durée doit être positive".to_string());
        }
        
        // Données contextuelles
        let mut data = HashMap::new();
        data.insert("simulation".to_string(), b"true".to_vec());
        data.insert("duration_secs".to_string(), duration_secs.to_string().into_bytes());
        
        // Cascade hormonale de stress
        
        // 1. Émission d'adrénaline (réaction immédiate)
        self.emit_hormone(
            HormoneType::Adrenaline,
            "stress_simulation",
            intensity,
            1.0,
            0.8,
            data.clone(),
        )?;
        
        // 2. Émission de cortisol (réaction secondaire)
        self.emit_hormone(
            HormoneType::Cortisol,
            "stress_simulation",
            intensity * 0.8,
            1.0,
            1.5,
            data.clone(),
        )?;
        
        // 3. Diminution de la sérotonine (effet négatif du stress)
        // Cette émission spéciale crée un "trou" hormonal pour la sérotonine
        let mut serotonin_data = data.clone();
        serotonin_data.insert("decrease".to_string(), b"true".to_vec());
        
        let current_serotonin = self.get_hormone_level(&HormoneType::Serotonin);
        if current_serotonin > 0.2 {
            self.hormone_levels.insert(HormoneType::Serotonin, current_serotonin * (1.0 - intensity * 0.5));
        }
        
        // 4. Augmentation de la noradrénaline (vigilance accrue)
        self.emit_hormone(
            HormoneType::Norepinephrine,
            "stress_simulation",
            intensity * 0.7,
            0.9,
            1.0,
            data,
        )?;
        
        Ok(())
    }
    
    /// Méthode pour simuler une réponse de bien-être/calme
    pub fn simulate_wellbeing_response(&self, intensity: f64) -> Result<(), String> {
        // Vérification des paramètres
        if intensity <= 0.0 || intensity > 1.0 {
            return Err("L'intensité doit être entre 0.0 et 1.0".to_string());
        }
        
        // Données contextuelles
        let mut data = HashMap::new();
        data.insert("simulation".to_string(), b"true".to_vec());
        data.insert("wellbeing".to_string(), b"true".to_vec());
        
        // Cascade hormonale de bien-être
        
        // 1. Émission de dopamine (récompense/plaisir)
        self.emit_hormone(
            HormoneType::Dopamine,
            "wellbeing_simulation",
            intensity * 0.8,
            0.9,
            1.0,
            data.clone(),
        )?;
        
        // 2. Émission de sérotonine (stabilité/bien-être)
        self.emit_hormone(
            HormoneType::Serotonin,
            "wellbeing_simulation",
            intensity,
            0.9,
            1.2,
            data.clone(),
        )?;
        
        // 3. Émission d'ocytocine (confiance/attachement)
        self.emit_hormone(
            HormoneType::Oxytocin,
            "wellbeing_simulation",
            intensity * 0.7,
            0.8,
            1.0,
            data.clone(),
        )?;
        
        // 4. Émission d'endorphines (bien-être physique)
        self.emit_hormone(
            HormoneType::Endorphin,
            "wellbeing_simulation",
            intensity * 0.6,
            0.7,
            0.9,
            data,
        )?;
        
        // 5. Réduction des hormones de stress
        let current_adrenaline = self.get_hormone_level(&HormoneType::Adrenaline);
        let current_cortisol = self.get_hormone_level(&HormoneType::Cortisol);
        
        self.hormone_levels.insert(HormoneType::Adrenaline, current_adrenaline * (1.0 - intensity * 0.3));
        self.hormone_levels.insert(HormoneType::Cortisol, current_cortisol * (1.0 - intensity * 0.2));
        
        Ok(())
    }

impl Default for HormonalField {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hormone_emission() {
        let field = HormonalField::new();
        
        // Émettre de l'adrénaline
        let result = field.emit_hormone(
            HormoneType::Adrenaline,
            "test",
            0.8,
            1.0,
            1.0,
            HashMap::new(),
        );
        
        assert!(result.is_ok());
        
        // Vérifier que le niveau a été mis à jour
        let level = field.get_hormone_level(&HormoneType::Adrenaline);
        assert!(level > 0.0);
    }
    
    #[test]
    fn test_receptor_activation() {
        let field = HormonalField::new();
        let activation_detected = Arc::new(Mutex::new(false));
        
        let activation_detected_clone = activation_detected.clone();
        
        // Enregistrer un récepteur de test
        field.register_receptor(
            "test_receptor",
            &[HormoneType::Dopamine],
            1.0,
            0.5,
            "test_owner",
            Box::new(move |_, _, _| {
                *activation_detected_clone.lock() = true;
                ReceptorAction::None
            }),
        ).unwrap();
        
        // Émettre de la dopamine au-dessus du seuil
        field.emit_hormone(
            HormoneType::Dopamine,
            "test",
            0.8,
            1.0,
            1.0,
            HashMap::new(),
        ).unwrap();
        
        // Activer les récepteurs
        field.activate_receptors();
        
        // Vérifier que le récepteur a été activé
        assert!(*activation_detected.lock());
    }
    
    #[test]
    fn test_hormone_degradation() {
        let field = HormonalField::new();
        
        // Émettre de la dopamine
        field.emit_hormone(
            HormoneType::Dopamine,
            "test",
            1.0,
            1.0,
            1.0,
            HashMap::new(),
        ).unwrap();
        
        let initial_level = field.get_hormone_level(&HormoneType::Dopamine);
        
        // Simuler une mise à jour avec un coefficient de dégradation
        field.update_hormone_levels(0.5);
        
        // Vérifier que le niveau a diminué
        let new_level = field.get_hormone_level(&HormoneType::Dopamine);
        assert!(new_level < initial_level);
    }
}
