mkdir -p datasets && \
cd datasets && \
find . -type d -name '__MACOSX' -exec rm -rf {} + && \
find . -name '.DS_Store' -delete && \
cd ..
