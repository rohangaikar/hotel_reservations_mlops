FROM jenkins/jenkins:lts

USER root

# First identify the actual Debian version
RUN cat /etc/os-release

# Update and install prerequisites
RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg

# Modern way to add Docker's GPG key (works for Debian 10+)
RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository - adjust "bookworm" based on your os-release output
RUN echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
RUN apt-get update && \
    apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin && \
    apt-get clean

# Rest of your original configuration...
RUN groupadd -f docker && \
    usermod -aG docker jenkins

USER jenkins
