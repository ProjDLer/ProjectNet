#!Use this script to tune parameters
use strict;

open(FOUT,">result.txt");

my@ use_tail_mats=("1");
my@ tail_ranks=("100", "90", "80", "70",  "60", "50");
my@ head_ranks=("50", "60", "70", "80", "90", "100");
my@ iters=("10");
my@ ratios=("0");
my@ lambdas=("0.1","0.5","1");

foreach my$use_tail_mat(@use_tail_mats)
	{
		foreach my$iter(@iters){
			foreach my$ratio(@ratios){
				foreach my$head_rank(@head_ranks){
					foreach my$tail_rank(@tail_ranks){
					foreach my$lambda(@lambdas){
						my$text="enwiki9_phrase";
						my$train="../W2V-Relation/Dataset/".$text;
						my$size=100;
						my $read_vocab = $train."_vocab.txt";
						my$binary=2;
						my$cbow=0;
						my$alpha=0.025;
						my$sample=0;
						my$hs=0;
						my$threads=9;
						my$mincount=5;
						my$negative=5;
						my$window=5;
						my$margin=1;
						my$use_relat=1;
						my$init_learning_rate = 0.025;
						my $relat_file = "../W2V-Relation/Dataset/valid_relats_Freebase13_train.txt";
						my $sw=1;
						my $sw_file = "../W2V-Relation/Dataset/stop_words_eh.txt";
						my $epoch=1;
						my $is_diag= 1;
						my $update_mat= 1;
						
						my $out_str="Output/028/".$text."_hrank".$head_rank."_trank".$tail_rank."_lambda".$lambda."_ratio".$ratio."_iter".$iter."_usetailmat".$use_tail_mat."updatemat".$update_mat."_diag".$is_diag;
						my $binary_embedding_file = $out_str.".bin";
						my $text_embedding_file = $out_str.".txt";
						my $text_out_layer_file = $out_str."_outputlayer.txt";
						my $bin_out_layer_file = $out_str."_outputlayer.bin";
						my $text_relat_file = $out_str."_relat.txt";
						my $bin_relat_file = $out_str."_relat.bin";
						
						my $args = "-update_mat ".$update_mat." -train ".$train." -use_tail_mat ".$use_tail_mat." -is_diag ".$is_diag." -update_ratio ".$ratio." -cbow ".$cbow." -head_relat_rank ".$head_rank." -tail_relat_rank ".$tail_rank." -binary_embedding_file ".$binary_embedding_file." -text_embedding_file ".$text_embedding_file." -out_relat_text ".$text_relat_file." -out_relat_binary ".$bin_relat_file." -threads ".$threads." -size ".$size." -binary ".$binary." -epoch ".$epoch." -negative ".$negative." -init_learning_rate ".$init_learning_rate." -hs ".$hs." -sample ".$sample." -mincount ".$mincount." -use_relation ".$use_relat." -window ".$window." -stopwords ".$sw." -sw_file ".$sw_file." -read_vocab ".$read_vocab." -relation_file ".$relat_file." -margin ".$margin." -lambda ".$lambda." -outputlayer_text_file ".$text_out_layer_file." -outputlayer_bin_file ".$bin_out_layer_file." -know_iter ".$iter;

						system("word2vec_projection.exe $args");
						
						my $result = `w2v_proj_analogical_evaluate.exe $binary_embedding_file ../W2V-Relation/Dataset/questions-fb13.txt 1 $bin_relat_file`;
						print $out_str." $result";
						print FOUT $out_str."\n $result";
			
					}
					}
				}
			}
		}
	}

	close(FOUT);