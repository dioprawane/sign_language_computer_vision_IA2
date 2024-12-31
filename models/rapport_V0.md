Rapport de classification :
                 precision    recall  f1-score   support

              0       0.67      0.90      0.77       418
              1       0.99      1.00      1.00       396
              2       0.99      1.00      1.00       404
              3       1.00      1.00      1.00       396
              4       0.99      0.98      0.99       399
              5       1.00      1.00      1.00       398
              6       1.00      0.99      0.99       420
              7       1.00      1.00      1.00       405
              8       1.00      1.00      1.00       401
              9       0.84      0.52      0.65       408
              A       1.00      0.97      0.99       381
              B       0.99      1.00      0.99       412
              C       0.99      0.95      0.97       397
              D       1.00      1.00      1.00       386
              E       0.99      1.00      0.99       406
              F       0.97      0.98      0.97       400
              G       0.99      1.00      1.00       436
              H       0.99      0.99      0.99       427
              I       1.00      1.00      1.00       389
              J       1.00      1.00      1.00       413
              K       0.99      1.00      1.00       396
              L       1.00      1.00      1.00       362
              M       0.99      0.99      0.99       392
              N       0.99      1.00      1.00       429
              O       0.95      1.00      0.97       409
              P       1.00      0.99      1.00       429
              Q       0.99      1.00      1.00       397
              R       0.92      0.95      0.94       401
              S       0.97      0.99      0.98       415
              T       1.00      0.99      0.99       418
              U       0.94      0.96      0.95       391
              V       0.99      0.93      0.96       354
              W       1.00      1.00      1.00       392
              X       1.00      1.00      1.00       412
              Y       1.00      1.00      1.00       410
              Z       1.00      0.99      1.00       394
          aimer       0.99      1.00      1.00       374
        appeler       1.00      0.99      1.00       393
       attraper       0.99      1.00      1.00       395
          coeur       1.00      1.00      1.00       382
    deux_coeurs       1.00      0.99      1.00       374
doigt_d_honneur       0.99      1.00      0.99       395
         espace       1.00      1.00      1.00       393
   ne_pas_aimer       1.00      1.00      1.00       394
             ok       1.00      0.99      0.99       375
          paume       1.00      1.00      1.00       400
         pierre       1.00      1.00      1.00       417
       pistolet       1.00      0.99      1.00       389
          point       0.99      1.00      0.99       400
  prendre_photo       0.99      1.00      0.99       396
         prière       0.99      0.95      0.97       409
           stop       1.00      1.00      1.00       422
     temps_mort       0.95      1.00      0.98       400

       accuracy                           0.98     21201
      macro avg       0.98      0.98      0.98     21201
   weighted avg       0.98      0.98      0.98     21201



   """Dense(1024, activation='relu', input_shape=(input_shape,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(96, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')"""