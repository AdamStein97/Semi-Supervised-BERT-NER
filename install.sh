mkdir "saved_models"

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qhfYhT1w4LrPGAIhZc9ObozcVUKmhC-H' -O saved_models/BERT_NER_confidence_kl_final.data-00000-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KiQNj4dOtQZNkIhXb720e4K5sWyYNHez' -O saved_models/BERT_NER_confidence_kl_final.data-00001-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=19BwzNiHzBLzLNwB3__tPrvB7PIFUGhWn' -O saved_models/BERT_NER_confidence_kl_final.index
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1d0qdrQQ2v-QG5xbnU8CFeAsbVBQzU2l9' -O saved_models/BERT_NER_data_dist_kl_final.data-00000-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zLlbYnnYs8vdw8jLq-UJJzdSaJcsjVaC' -O saved_models/BERT_NER_data_dist_kl_final.data-00001-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Nxiy4ODnfh2S-19NbgB9qivDvVhLc6lL' -O saved_models/BERT_NER_data_dist_kl_final.index
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c3xtW55V9-fPkjl-tzOLziosKJvoGqVh' -O saved_models/BERT_NER_final.data-00000-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wCql1xxksLtVxMhzfK031Nngvw98vlUB' -O saved_models/BERT_NER_final.data-00001-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=18FWkLmbkX-rJGojzRJuvK_dWD_MwlpQm' -O saved_models/BERT_NER_final.index
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PORdTKbDQZX196jQzunECDXwyM4SOIGG' -O saved_models/NER_baseline_final.data-00000-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1vNm-AhOy1ivyT9BxJ0ktWIvPODy5cXRf' -O saved_models/NER_baseline_final.data-00001-of-00002
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1U7t-mdEO9BAm1ShrMPyL9IGuLD1HgCKl' -O saved_models/NER_baseline_final.index


pip install -r requirements.txt