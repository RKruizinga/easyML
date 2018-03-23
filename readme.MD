python3 main.py --data_folder=data/gender_age_prediction/ --predict_label=age --predict_languages=e --method=svm --k=1 --print_details=5


python main.py --data_folder=data/adsearchreport/ --predict_label=ctr --method=svm --k=1 --print_details=1 --predict_method=regression


--- for conversions of a particular client
python conversion_chance.py --data_folder=data/client_scores/pme/ --predict_label=conversion_chance --method=svm --k=1 --print_details=1 --predict_method=classification --avoid_skewness=True

python conversion_path.py --data_folder=data/client_scores/pme/ --predict_label=conversion_path --method=svm --k=1 --print_details=1 --predict_method=classification --avoid_skewness=True

python conversion_product.py --data_folder=data/client_scores/pme/ --predict_label=conversion_product --method=svm --k=1 --print_details=1 --predict_method=classification --avoid_skewness=True