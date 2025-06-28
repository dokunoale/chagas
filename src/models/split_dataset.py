from sklearn.model_selection import train_test_split

def split_dataset(X, y, test_size=0.2, val_size=0.25, random_state=42):
    """
    Divide il dataset in training, validation e test set.
    
    Parametri:
    - X: feature matrix
    - y: target array
    - test_size: frazione del dataset da usare come test (default: 0.2)
    - val_size: frazione del train_val da usare come validation (default: 0.25)
    - random_state: per riproducibilità (default: 42)
    
    Ritorna:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Split in train+val e test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split in train e val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test