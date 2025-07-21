pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = 'compact-arc-464706-r2'
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
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
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Buidling and Pushing Docker Image to GCR'){
            steps{
                withCredentials([file(credentialsId :'gcp-key', variable:'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script {
                        echo 'Buidling and Pushing Docker Image to GCR.....'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project ${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest

                      '''
                    
                    }
                }
            }
        }
    }
} 