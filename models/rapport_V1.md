Rapport de classification :
                 precision    recall  f1-score   support

              0       0.55      1.00      0.71       418
              1       0.99      1.00      1.00       396
              2       0.97      1.00      0.99       404
              3       1.00      1.00      1.00       396
              4       0.99      0.99      0.99       399
              5       1.00      1.00      1.00       398
              6       1.00      0.99      0.99       420
              7       1.00      0.98      0.99       405
              8       0.97      1.00      0.98       401
              9       0.73      0.18      0.29       408
              A       0.85      0.99      0.92       381
              B       0.99      0.99      0.99       412
              C       0.86      0.95      0.90       397
              D       1.00      0.97      0.98       386
              E       0.97      0.93      0.95       406
              F       1.00      0.98      0.99       400
              G       0.96      1.00      0.98       436
              H       1.00      0.99      0.99       427
              I       1.00      1.00      1.00       389
              J       0.96      1.00      0.98       413
              K       0.99      0.99      0.99       396
              L       1.00      1.00      1.00       362
              M       0.89      0.99      0.94       392
              N       1.00      0.90      0.94       429
              O       1.00      0.86      0.92       409
              P       0.97      1.00      0.98       429
              Q       1.00      0.99      0.99       397
              R       1.00      0.30      0.46       401
              S       0.99      0.82      0.90       415
              T       1.00      0.95      0.97       418
              U       0.60      0.99      0.75       391
              V       0.98      0.83      0.90       354
              W       0.85      1.00      0.92       392
              X       0.98      0.96      0.97       412
              Y       0.99      1.00      0.99       410
              Z       1.00      0.98      0.99       394
          aimer       1.00      1.00      1.00       374
        appeler       1.00      1.00      1.00       393
       attraper       1.00      0.52      0.68       395
          coeur       1.00      0.63      0.77       382
    deux_coeurs       0.70      0.90      0.78       374
doigt_d_honneur       1.00      0.98      0.99       395
         espace       0.99      1.00      0.99       393
   ne_pas_aimer       1.00      0.98      0.99       394
             ok       0.89      0.99      0.94       375
          paume       0.69      1.00      0.82       400
         pierre       0.96      1.00      0.98       417
       pistolet       1.00      1.00      1.00       389
          point       0.97      0.99      0.98       400
  prendre_photo       0.99      0.99      0.99       396
         prière       0.98      0.99      0.98       409
           stop       0.98      1.00      0.99       422
     temps_mort       0.98      0.97      0.98       400

       accuracy                           0.93     21201
      macro avg       0.95      0.93      0.93     21201
   weighted avg       0.95      0.93      0.93     21201


   model = Sequential([
        Dense(1024, activation='relu', input_shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(96, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])