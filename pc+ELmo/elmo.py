import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext import vocab
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import numpy as np
import requests
import zipfile
import hashlib
import time
import sys

# Download NLTK tokenizer data
nltk.download('punkt', quiet=True)

# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def preprocess_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    
    # Tokenize the text into words
    words = word_tokenize(str(text).lower())
    
    # List of prepositions to remove (customize as needed)
    prepositions = ['for', 'with', 'to', 'from', 'in', 'on', 'at', 'by']
    
    # Filter out prepositions and specific characters from words
    filtered_words = [word for word in words if word not in prepositions and word.isalnum()]
    
    # Join filtered words back into a cleaned text string
    cleaned_text = ' '.join(filtered_words)
    
    return cleaned_text

def clean_dataframe(df, column_name):
    # Apply preprocess_text function to the specified column in the DataFrame
    df[column_name] = df[column_name].apply(preprocess_text)
    return df

# Alternative approach - skip downloading GloVe and create random embeddings for testing
def create_random_embeddings(vocab_obj, embedding_dim=300):
    """Create random embeddings for vocabulary words."""
    print("Creating random embeddings for testing...")
    vocab_size = len(vocab_obj)
    # Initialize with a fixed seed for reproducibility
    np.random.seed(42)
    return np.random.randn(vocab_size, embedding_dim) * 0.1  # Small values to start

def download_file_with_progress(url, destination):
    """Download a file with progress reporting."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        print(f"Downloading from {url}")
        print(f"File size: {total_size / (1024*1024):.2f} MB")
        
        # Progress bar setup
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        # Open file and start writing chunks
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        # Verify download completed properly
        if total_size != 0 and t.n != total_size:
            print("Downloaded size does not match expected size!")
            return False
        
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(destination):
            os.remove(destination)  # Remove potentially corrupted file
        return False

def verify_glove_files(embedding_dim=300):
    """Manually load and verify GloVe files without using ZIP."""
    # Create vectors directory
    vectors_dir = os.path.join(os.getcwd(), "glove.6B")
    os.makedirs(vectors_dir, exist_ok=True)
    
    # Direct download URLs for individual GloVe files (replacing the ZIP download)
    glove_urls = {
        50: "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.50d.txt",
        100: "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.100d.txt",
        200: "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.200d.txt",
        300: "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.300d.txt"
    }
    
    # Choose the URL based on embedding dimension
    if embedding_dim not in glove_urls:
        print(f"No GloVe embeddings available for dimension {embedding_dim}")
        print(f"Available dimensions: {list(glove_urls.keys())}")
        print(f"Defaulting to 300d embeddings")
        embedding_dim = 768
    
    glove_url = glove_urls[embedding_dim]
    file_path = os.path.join(vectors_dir, f"glove.6B.{embedding_dim}d.txt")
    
    # Download if file doesn't exist or is empty
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:
        print(f"Downloading GloVe {embedding_dim}d embeddings...")
        success = download_file_with_progress(glove_url, file_path)
        
        if not success:
            print("Download failed. Using random embeddings instead.")
            return None
        
        # Verify file content (checking first few lines)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line or len(first_line.split()) <= 1:
                    print("Downloaded file appears to be invalid.")
                    return None
        except Exception as e:
            print(f"Error verifying GloVe file: {e}")
            return None
    
    return file_path

def load_glove_embeddings(embeddings_file, embedding_dim=300):
    """Load GloVe embeddings from file."""
    if embeddings_file is None:
        return None
        
    print(f"Loading GloVe embeddings from {embeddings_file}...")
    embeddings_dict = {}
    try:
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.strip().split()
                if len(values) != embedding_dim + 1:
                    continue  # Skip malformed lines
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_dict[word] = vector
        
        print(f"Loaded {len(embeddings_dict)} word vectors")
        return embeddings_dict
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None

def create_embedding_matrix(vocab_obj, embeddings_dict=None, embedding_dim=300):
    """Create embedding matrix from vocabulary and embeddings dictionary."""
    vocab_tokens = vocab_obj.get_itos()
    embed_matrix = np.zeros((len(vocab_tokens), embedding_dim))
    
    # If no embeddings dictionary is provided, return random embeddings
    if embeddings_dict is None:
        print("Using random embeddings")
        return np.random.randn(len(vocab_tokens), embedding_dim) * 0.1
    
    found_words = 0
    for ind, word in enumerate(vocab_tokens):
        if word in embeddings_dict:
            embed_matrix[ind] = embeddings_dict[word]
            found_words += 1
    
    print(f"Found embeddings for {found_words}/{len(vocab_tokens)} words ({found_words/len(vocab_tokens)*100:.2f}%)")
    return embed_matrix

# Load and prepare the dataset
filepath = "/Users/shriya/Documents/GitHub/logo_detect/dsmlprojects/pc+ELmo/vasavi2.csv"
num_rows_to_read = 46

class ProcessYelp():
    def __init__(self, cleaned_df, min_freq):
        self.min_freq = min_freq

        total_words = []
        for i in tqdm(range(len(cleaned_df)), desc="Building Vocabulary"):
            if 'DESCRIPTION' in cleaned_df.columns and not pd.isna(cleaned_df['DESCRIPTION'][i]):
                line = str(cleaned_df['DESCRIPTION'][i])
                total_words += [[word.lower()] for word in word_tokenize(line)]

        # Build vocabulary from tokenized words
        self.vocab = vocab.build_vocab_from_iterator(total_words,
                                                    min_freq=min_freq,
                                                    specials=['<UNK>', '<PAD>'])
        self.vocab.set_default_index(self.vocab['<UNK>'])
        
        print(f"Vocabulary size: {len(self.vocab)}")

class LabelData(Dataset):
    def __init__(self, vocab, cleaned_df, max_length=35):
        self.vocab = vocab
        self.data = cleaned_df
        self.max_length = max_length

    def __getitem__(self, index):
        description = str(self.data.loc[index, 'DESCRIPTION']) if not pd.isna(self.data.loc[index, 'DESCRIPTION']) else ""
        label = self.data.loc[index, 'Style name encoded']

        # Tokenize the description
        tokens = word_tokenize(description.lower())
        
        # Convert tokens to indices using vocab, and pad if necessary
        token_indices = [self.vocab[token.lower()] if token.lower() in self.vocab else self.vocab['<UNK>'] for token in tokens[:self.max_length]]
        token_indices += [self.vocab['<PAD>']] * (self.max_length - len(token_indices))

        return torch.tensor(token_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, embeddings):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)

        self.layer_1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.layer_2 = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, X):
        embeddings = self.embedding(X)
        lstm1_output, _ = self.layer_1(embeddings)
        lstm2_output, _ = self.layer_2(lstm1_output)
        lstm2_output = self.dropout(lstm2_output)
        output = self.linear(lstm2_output)
        output = torch.transpose(output, 1, 2)
        return output

def main():
    try:
        # Load the dataset
        print(f"Loading dataset from: {filepath}")
        df = pd.read_csv(filepath, nrows=num_rows_to_read)
        
        # Specify the column containing text to be cleaned
        text_columns = ['DESCRIPTION', 'FABRIC DESCRIPTION']
        for text_column in text_columns:
            if text_column in df.columns:
                df = clean_dataframe(df, text_column)
        
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Process the data
        processed_data = ProcessYelp(df, min_freq=5)
        
        # Create dataset and dataloader
        label_train_dataset = LabelData(processed_data.vocab, df, max_length=35)
        
        # Example: access a single item
        index = 2 if len(label_train_dataset) > 2 else 0
        tokens, label = label_train_dataset[index]
        print(f"Example data point: tokens shape={tokens.shape}, label={label.item()}")
        
        # Create dataloader
        BATCH_SIZE = 64
        train_dataloader = DataLoader(label_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Set embedding dimension
        embedding_dim = 300
        
        # Try to get GloVe embeddings
        glove_file = verify_glove_files(embedding_dim)
        embeddings_dict = load_glove_embeddings(glove_file, embedding_dim)
        
        # Create embedding matrix (either from GloVe or random if GloVe failed)
        embed_matrix = create_embedding_matrix(processed_data.vocab, embeddings_dict, embedding_dim)
        
        # Convert to torch tensor
        embed_matrix_torch = torch.tensor(embed_matrix, dtype=torch.float)
        embed_matrix_torch = embed_matrix_torch.to(DEVICE)
        
        # Initialize model and training parameters
        hidden_size = 150
        dropout = 0.1
        learning_rate = 0.001
        epochs = 1
        
        # Create model
        elmo = ELMo(
            len(processed_data.vocab), 
            embedding_dim, 
            hidden_size, 
            dropout, 
            embed_matrix_torch
        ).to(DEVICE)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        optimizer = optim.Adam(elmo.parameters(), lr=learning_rate)
        
        # Training loop
        print("Starting training...")
        elmo.train()
        for epoch in range(epochs):
            total_train_loss = 0
            for sentence, label in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inp = sentence[:, :-1].to(DEVICE)
                targ = sentence[:, 1:].to(DEVICE)
                
                optimizer.zero_grad()
                output = elmo(inp)
                loss = criterion(output, targ)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_loss = total_train_loss / len(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save the model
        output_path = 'bilstm2.pt'
        torch.save(elmo.state_dict(), output_path)
        print(f"Model saved to {output_path}")
        print("ELMo embeddings successfully created!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()