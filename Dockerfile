FROM ubuntu:20.04

LABEL maintainer="TCC Feature Engineering Pipeline"
LABEL description="Environment for Defects4J feature extraction with Java 11, Perl, and Python"

# Evitar prompts interativos durante instalação
ENV DEBIAN_FRONTEND=noninteractive

#############################################################################
# Instalar dependências base
#############################################################################

RUN apt-get update -y && \
    apt-get install -y \
    openjdk-11-jdk \
    git \
    build-essential \
    subversion \
    perl \
    curl \
    unzip \
    cpanminus \
    make \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Configurar Java 11
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Configurar timezone (requerido pelo Defects4J)
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#############################################################################
# Setup Defects4J
#############################################################################

WORKDIR /defects4j

# Copiar Defects4J local (já baixado, mas SEM project_repos)
COPY defects4j /defects4j

# Corrigir line endings (Windows → Unix) em TODOS os arquivos texto do Defects4J
RUN apt-get update && apt-get install -y dos2unix file && \
    echo "Convertendo line endings de todos os scripts..." && \
    find /defects4j -type f -exec sh -c 'file "$1" | grep -q "text" && dos2unix "$1" 2>/dev/null || true' _ {} \; && \
    echo "Instalando dependências Perl..." && \
    cpanm --installdeps . && \
    echo "Executando init.sh..." && \
    chmod +x ./init.sh ./project_repos/get_repos.sh && \
    ./init.sh && \
    echo "Verificando project_repos..." && \
    ls -la /defects4j/project_repos/ | head -20 && \
    rm -rf /var/lib/apt/lists/*

# Adicionar Defects4J ao PATH
ENV PATH="/defects4j/framework/bin:${PATH}"

# Validar instalação do Defects4J
RUN defects4j info -p Lang || echo "Defects4J validation check"

#############################################################################
# Setup Python e Dependências do TCC
#############################################################################

WORKDIR /app

# Copiar requirements.txt primeiro (cache do Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copiar código do TCC
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/

# Criar diretórios necessários
RUN mkdir -p data/raw data/intermediate data/processed data/results logs

#############################################################################
# Configuração de execução
#############################################################################

# Expor volumes para persistência
VOLUME ["/app/data", "/app/logs"]

# Comando padrão: executar o pipeline
CMD ["python3", "scripts/prepare_dataset.py"]

