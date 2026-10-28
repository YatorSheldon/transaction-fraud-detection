// CI/CD pipeline: lint -> test -> train -> quality gate -> build image -> approval -> deploy.
pipeline {
    agent any

    environment {
        VENV = "${WORKSPACE}/.venv"
        IMAGE = "fraudguard-api:${BUILD_NUMBER}"
        MIN_PR_AUC = "0.40"   // the model must clear this on the test period to ship
    }

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install --quiet --upgrade pip
                    pip install --quiet -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '. "$VENV/bin/activate" && ruff check . && ruff format --check .'
            }
        }

        stage('Unit tests') {
            steps {
                sh '. "$VENV/bin/activate" && pytest --junitxml=reports/junit.xml'
            }
            post {
                always { junit 'reports/junit.xml' }
            }
        }

        stage('Train model') {
            steps {
                sh '. "$VENV/bin/activate" && PYTHONPATH=src python -m fraudguard.train'
            }
        }

        stage('Model quality gate') {
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    python - <<'EOF'
import json, os, sys
m = json.load(open("reports/metrics.json"))
best = m["models"][m["best_model"]]
print(f"Best model: {m['best_model']}  PR-AUC={best['pr_auc']}  recall={best['recall']}  precision={best['precision']}")
if best["pr_auc"] < float(os.environ["MIN_PR_AUC"]):
    sys.exit(f"PR-AUC {best['pr_auc']} is below the gate of {os.environ['MIN_PR_AUC']}")
EOF
                '''
            }
        }

        stage('Build API image') {
            steps {
                sh 'docker build -t "$IMAGE" .'
            }
        }

        stage('Approval') {
            when { branch 'main' }
            steps {
                input message: 'Deploy this model to production?', ok: 'Deploy'
            }
        }

        stage('Deploy') {
            when { branch 'main' }
            steps {
                sh '''
                    docker rm -f fraudguard-api || true
                    docker run -d --name fraudguard-api -p 8000:8000 "$IMAGE"
                    sleep 5
                    curl --fail http://localhost:8000/health
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'reports/**', allowEmptyArchive: true
        }
        success { echo 'Pipeline passed: model trained, gated and deployed.' }
        failure { echo 'Pipeline failed: check the stage logs above.' }
    }
}
