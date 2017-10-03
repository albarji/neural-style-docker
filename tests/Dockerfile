FROM albarji/neural-style

# Add testing tools
RUN conda install -y \
    nose \
    coverage \
    && conda clean -y -i -l -p -t

# Add tests and test images
WORKDIR /app/entrypoint
COPY ["*.py", "/app/entrypoint/tests/"]
COPY ["contents/*", "/app/entrypoint/tests/contents/"]
COPY ["styles/*", "/app/entrypoint/tests/styles/"]

# Entrypoint runs tests
ENTRYPOINT nosetests --with-coverage --cover-package=neuralstyle -v
