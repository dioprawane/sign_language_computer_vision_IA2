Rapport de classification :
                 precision    recall  f1-score   support

              0       0.92      0.72      0.81       418
              1       0.98      1.00      0.99       396
              2       1.00      1.00      1.00       404
              3       1.00      0.99      0.99       396
              4       1.00      0.98      0.99       399
              5       1.00      1.00      1.00       398
              6       1.00      1.00      1.00       420
              7       0.99      1.00      1.00       405
              8       1.00      1.00      1.00       401
              9       0.77      0.94      0.85       408
              A       1.00      0.97      0.99       381
              B       0.99      1.00      0.99       412
              C       1.00      1.00      1.00       397
              D       1.00      1.00      1.00       386
              E       1.00      0.99      0.99       406
              F       1.00      0.95      0.97       400
              G       1.00      1.00      1.00       436
              H       1.00      0.97      0.99       427
              I       0.99      1.00      1.00       389
              J       0.98      0.98      0.98       413
              K       1.00      0.99      0.99       396
              L       1.00      1.00      1.00       362
              M       0.99      0.86      0.92       392
              N       0.89      1.00      0.94       429
              O       1.00      1.00      1.00       409
              P       1.00      0.99      1.00       429
              Q       1.00      0.99      0.99       397
              R       0.95      1.00      0.97       401
              S       0.96      0.99      0.98       415
              T       0.99      0.99      0.99       418
              U       0.99      0.96      0.97       391
              V       0.98      0.97      0.98       354
              W       1.00      1.00      1.00       392
              X       0.98      1.00      0.99       412
              Y       1.00      0.99      1.00       410
              Z       0.99      0.97      0.98       394
          aimer       0.86      1.00      0.92       374
        appeler       1.00      0.98      0.99       393
       attraper       1.00      1.00      1.00       395
          coeur       0.99      1.00      0.99       382
    deux_coeurs       1.00      1.00      1.00       374
doigt_d_honneur       0.98      0.99      0.99       395
         espace       1.00      1.00      1.00       393
   ne_pas_aimer       1.00      1.00      1.00       394
             ok       0.98      1.00      0.99       375
          paume       1.00      1.00      1.00       400
         pierre       0.99      1.00      0.99       417
       pistolet       1.00      1.00      1.00       389
          point       0.98      0.98      0.98       400
  prendre_photo       1.00      0.85      0.92       396
         prière       0.96      0.99      0.97       409
           stop       1.00      1.00      1.00       422
     temps_mort       0.99      0.98      0.99       400

       accuracy                           0.98     21201
      macro avg       0.98      0.98      0.98     21201
   weighted avg       0.98      0.98      0.98     21201



   model = Sequential([
        Dense(512, activation='relu', input_shape=(input_shape,)),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(96, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])