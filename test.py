

base_model = xception.Xception(include_top=False, weights='imagenet', input_shape=(299, 299,3))
x = base_model.output

x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(196, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(keep_checkpoint_max=5, save_summary_steps=10)
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy', metrics.top_k_categorical_accuracy, metrics.mean_absolute_error, auc])
est = tf.keras.estimator.model_to_estimator(model, model_dir='out', config=run_config)

