//! Database management for settings and persistence.

use std::path::Path;

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use serde_json; // Added for `serde_json::from_str` and `serde_json::to_string`

pub struct Database {
    conn: Connection,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ModelSettings {
    #[serde(default)]
    pub values: std::collections::HashMap<String, serde_json::Value>,
}

impl Database {
    pub fn new(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }
    fn init_schema(&self) -> anyhow::Result<()> {
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )",
            [],
        )?;

        // Drop old table and create new relational one
        self.conn.execute("DROP TABLE IF EXISTS model_settings", [])?;
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS model_setting_values (
                model_id TEXT,
                key TEXT,
                value_json TEXT,
                PRIMARY KEY (model_id, key)
            )",
            [],
        )?;

        Ok(())
    }

    // --- App Settings ---

    pub fn get_app_setting(&self, key: &str) -> anyhow::Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM app_settings WHERE key = ?")?;
        let mut rows = stmt.query(params![key])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub fn set_app_setting(&self, key: &str, value: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
            params![key, value],
        )?;
        Ok(())
    }

    // --- Model Settings ---

    pub fn get_model_settings(&self, model_id: &str) -> anyhow::Result<ModelSettings> {
        let mut stmt = self
            .conn
            .prepare("SELECT key, value_json FROM model_setting_values WHERE model_id = ?")?;
        let rows = stmt.query_map(params![model_id], |row| {
            let key: String = row.get(0)?;
            let json: String = row.get(1)?;
            Ok((key, json))
        })?;

        let mut values = std::collections::HashMap::new();
        for row in rows {
            let (key, json) = row?;
            if let Ok(value) = serde_json::from_str(&json) {
                values.insert(key, value);
            }
        }

        Ok(ModelSettings { values })
    }

    pub fn set_model_settings(&self, model_id: &str, settings: &ModelSettings) -> anyhow::Result<()> {
        for (key, value) in &settings.values {
            let json = serde_json::to_string(value)?;
            self.conn.execute(
                "INSERT INTO model_setting_values (model_id, key, value_json) 
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(model_id, key) DO UPDATE SET value_json = excluded.value_json",
                params![model_id, key, &json],
            )?;
        }
        Ok(())
    }
}
