def train_data_generator(purpose):    """Training data generator    
    :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)    
   """
    _x = np.zeros((FLAGS.batch_size, FLAGS.img_w, FLAGS.img_h, FLAGS.img_c), dtype=np.float)
    _y = np.zeros(FLAGS.batch_size, dtype=np.float)
    out_idx = 0
    ## Preconditions
    # purpose = 'train'
    assert len(imgs[purpose]) == len(wheels[purpose])
    n_purpose = len(imgs[purpose])    assert n_purpose > 0   
    while 1:        """Loading random frame of the video repeatly 
        """
        ## Get a random line and get the steering angle
        frame_idx = np.random.randint(n_purpose)        ## Find angle
        angle = wheels[purpose][frame_idx]        ## Find frame
        img = imgs[purpose][frame_idx]        ## Implement data augmentation
        # img, angle = data_augment_pipeline(img, angle)

        # Check if we've got valid values
        if img is not None:
            _x[out_idx] = img
            _y[out_idx] = angle
            out_idx += 1
        # Check if we've enough values to yield
        if out_idx >= FLAGS.batch_size:            yield _x, _y            # Reset the values back
            _x = np.zeros((FLAGS.batch_size, FLAGS.img_w, FLAGS.img_h, FLAGS.img_c), dtype=np.float)
            _y = np.zeros(FLAGS.batch_size, dtype=np.float)
            out_idx = 0

model.fit_generator(
        train_data_generator('train'),
        samples_per_epoch=FLAGS.train_batch_per_epoch * FLAGS.batch_size,
        nb_epoch=10,
        validation_data=val_data_generator('val'),
        nb_val_samples=FLAGS.batch_size,
        verbose=1
 )
