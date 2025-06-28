
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow((1 - p_t), gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)
    return focal_loss_fixed


def make_callback(name):
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max')
    checkpoint = ModelCheckpoint(f"{name}_best_model.h5",
                             monitor='val_auc',
                             mode='max',
                             save_best_only=True,
                             verbose=1)
    
    callback = [early_stop, reduce_lr, checkpoint]
    return callback



