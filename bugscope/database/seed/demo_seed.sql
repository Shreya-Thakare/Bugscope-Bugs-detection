-- BugScope demo seed data

INSERT INTO projects (id, name, repository_url) VALUES
(1, 'Payments Core', 'https://github.com/acme/payments-core'),
(2, 'Customer Portal', 'https://github.com/acme/customer-portal'),
(3, 'Data Pipeline', 'https://github.com/acme/data-pipeline'),
(4, 'Legacy Billing Adapter', 'https://github.com/acme/billing-adapter'),
(5, 'Identity Service', 'https://github.com/acme/identity-svc'),
(6, 'Mobile BFF', 'https://github.com/acme/mobile-bff');

INSERT INTO modules (id, project_id, name, language, module_type, loc, cyclomatic_complexity, code_churn, review_coverage, test_coverage) VALUES
(1001, 1, 'SettlementEngine', 'Java', 'Core', 4820, 41.0, 72.0, 0.380, 0.410),
(1002, 1, 'LedgerWriter', 'Java', 'Data', 2100, 22.0, 35.0, 0.610, 0.550),
(1003, 2, 'CheckoutWizard', 'JavaScript', 'UI', 960, 11.0, 18.0, 0.840, 0.790),
(1004, 4, 'CobolBridge', 'C++', 'Core', 6200, 58.0, 90.0, 0.220, 0.180),
(1005, 3, 'IngestKafkaWorker', 'Python', 'API', 1340, 15.0, 28.0, 0.700, 0.660),
(1006, 5, 'TokenIssuer', 'C#', 'Core', 780, 8.0, 9.0, 0.910, 0.880),
(1007, 1, 'FraudRules', 'Java', 'Core', 3050, 29.0, 44.0, 0.490, 0.520),
(1008, 6, 'ProfileAggregator', 'Java', 'API', 1120, 13.0, 16.0, 0.760, 0.720),
(1009, 2, 'AuthRedirect', 'JavaScript', 'API', 420, 6.0, 7.0, 0.930, 0.900),
(1010, 3, 'TransformSparkJob', 'Python', 'Data', 2680, 24.0, 51.0, 0.450, 0.400),
(1011, 4, 'RateTableParser', 'C++', 'Utility', 1890, 19.0, 33.0, 0.400, 0.370),
(1012, 5, 'SessionStore', 'C#', 'Data', 540, 5.0, 4.0, 0.950, 0.920);

INSERT INTO predictions (module_id, defect_probability, risk_band, defective_flag, model_mode) VALUES
(1001, 0.8100, 'High', 1, 'ml'),
(1002, 0.4800, 'Medium', 0, 'ml'),
(1003, 0.1900, 'Low', 0, 'ml'),
(1004, 0.9200, 'High', 1, 'ml'),
(1005, 0.4100, 'Medium', 0, 'ml'),
(1006, 0.1100, 'Low', 0, 'ml'),
(1007, 0.6700, 'High', 1, 'ml'),
(1008, 0.2400, 'Low', 0, 'ml'),
(1009, 0.0900, 'Low', 0, 'ml'),
(1010, 0.7300, 'High', 1, 'ml'),
(1011, 0.5600, 'Medium', 1, 'ml'),
(1012, 0.0700, 'Low', 0, 'ml');

INSERT INTO alerts (module_id, description, severity, status) VALUES
(1004, 'Predicted defect probability 92%. Extremely high complexity and low review coverage.', 'High', 'Open'),
(1001, 'LOC > 4k with cyclomatic complexity 41. Recommend split and targeted tests.', 'High', 'Open'),
(1010, 'Churn spike + review coverage below gate (45%).', 'High', 'Acknowledged'),
(1007, 'Process risk elevated; team experience and test coverage lag peers.', 'Medium', 'Open'),
(1011, 'Medium band with upward trend over last 3 scores.', 'Medium', 'Open'),
(1006, 'Previous false positive cleared after coverage improvement.', 'Low', 'Closed');
