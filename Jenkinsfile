pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Github Repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning Github Repo to Jenkins......'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/rohangaikar/hotel_reservations_mlops.git']])
                }
            }
        }

        stage('Setting up Virtual Environment & Installing Dependencies'){
            steps{
                script{
                    echo 'Setting up Virtual Environment & Installing Dependencies......'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . {VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
} 