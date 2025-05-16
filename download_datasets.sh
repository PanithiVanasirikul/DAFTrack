mkdir -p datasets && \
cd datasets && \
wget https://www.bu.edu/vip/files/CEPDOF.zip && \
unzip CEPDOF.zip && \
wget https://www.bu.edu/vip/files/WEPDTOF.zip && \
unzip WEPDTOF.zip && \
find . -type d -name '__MACOSX' -exec rm -rf {} + && \
find . -name '.DS_Store' -delete && \
cd ..
