-- BugScope schema (MySQL-compatible)

CREATE TABLE projects (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(150) NOT NULL,
  repository_url VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE modules (
  id INT AUTO_INCREMENT PRIMARY KEY,
  project_id INT NOT NULL,
  name VARCHAR(200) NOT NULL,
  language VARCHAR(50),
  module_type VARCHAR(50),
  loc INT,
  cyclomatic_complexity DECIMAL(8,2),
  code_churn DECIMAL(8,2),
  review_coverage DECIMAL(5,3),
  test_coverage DECIMAL(5,3),
  FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  module_id INT NOT NULL,
  predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  defect_probability DECIMAL(6,4),
  risk_band VARCHAR(20),
  defective_flag TINYINT,
  model_mode VARCHAR(30),
  FOREIGN KEY (module_id) REFERENCES modules(id)
);

CREATE TABLE alerts (
  id INT AUTO_INCREMENT PRIMARY KEY,
  module_id INT,
  description TEXT NOT NULL,
  severity VARCHAR(20) DEFAULT 'Medium',
  status VARCHAR(30) DEFAULT 'Open',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (module_id) REFERENCES modules(id)
);

CREATE INDEX idx_predictions_module ON predictions(module_id, predicted_at);
CREATE INDEX idx_alerts_status ON alerts(status);
