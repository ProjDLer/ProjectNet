#pragma once

#include "Dictionary.h"
#include <concurrent_unordered_map.h>
#include <concurrent_vector.h>
#include <fstream>
#include "util.h"
#include <time.h>

extern Dictionary* g_dict;


struct StruRelation
{
	int w1, w2, r;
	StruRelation(int _w1 = 0, int _w2 = 0, int _r = 0){
		w1 = _w1;
		w2 = _w2;
		r = _r;
	}
};

bool cmp_by_first_entity(StruRelation x, StruRelation y);

bool cmp_by_second_entity(StruRelation x, StruRelation y);

bool myequal(StruRelation x, StruRelation y);

class CRelation
{
public:
	CRelation();
	~CRelation();
	Concurrency::concurrent_vector <StruRelation> list;
	Concurrency::concurrent_vector <StruRelation> list_reverse;

	Concurrency::concurrent_vector <StruRelation> list_in_know_space;
	Concurrency::concurrent_vector <StruRelation> list_reverse_in_know_space;

	Concurrency::concurrent_unordered_map<int, std::pair<int, int>> head_idx_range_table; // For each relation r, head_idx_range_table[r] records the index range of relation tuples in which r is a head in list
	Concurrency::concurrent_unordered_map<int, std::pair<int, int>> tail_idx_range_table; //For each relation r, tail_idx_range_table[r] records the index range of relation tuples in which r is a tail in list_reverse
	Concurrency::concurrent_unordered_map<std::string, int> relation_name2id_table;
	Concurrency::concurrent_unordered_map<int, std::string> relation_id2name_table;

	Concurrency::concurrent_unordered_map<int, long long> entityid2wordid_table;
	Concurrency::concurrent_unordered_map<long long, int> wordid2entityid_table;

	int relat_size; //#relation
	int entity_size; //#entity
	int update_know_cnt;
	int* relat_triple_size; 
	real* relat_head_left_matrix;
	real* relat_head_right_matrix;
	real* relat_tail_left_matrix;
	real* relat_tail_right_matrix;

	real* relat_actual_left_matrix;
	real* relat_actual_right_matrix;
	
	real** p_head_left_mat, **p_head_right_mat, **p_tail_left_mat, **p_tail_right_mat;
	real **p_relat_emb, **p_relat_act_emb;
	real** p_actual_left_mat, **p_actual_right_mat;

	real* relat_emb_vecs;
	real* relat_actual_emb_vecs; //the 2\sigmoid(r_embedding)-1

	std::map<int, real> head_diag_mat_ele[MAX_RELAT_CNT];
	std::map<int, real> tail_diag_mat_ele[MAX_RELAT_CNT];

	long long sample_num;
	void LoadListFromFile(const char * fname);
	void BuildTableFromList();
	void InitKnowledgeEmb();
	void SaveRelationEmb();
	bool ContainsEntity(int word_id);
	inline void ConstrainParameters(int r);
	real SeqTrainRelation();
	bool know_lock;
	real CRelation::ComputeLoss(int head, int tail, int r);

	int SampleWordIdx();
	int SampleRelatIdx();
	
	void TrainRelation(int centerWord);

	void ComputeGradient(int label, real weight, int relat, real* grads_tmp, real* head_grads, real* tail_grads, real* relat_grads,
		real* head_left_mat_grads, real* head_right_mat_grads, real* tail_left_mat_grads, real* tail_right_mat_grads, 
		real* off_vec, real* Qh_h, real* Qt_t, real* PhT_off_vec, real* PtT_off_vec, real* head_embedding, real* tail_embedding,
		real* diag_head_mat_grads, real* diag_tail_mat_grads);

	void TrainRelatTriple(int head, bool head_is_word, int r, int tail);

	real ComputeScore(int head, int r, int tail, real* Qh_h, real* Qt_t, real* off_vec);

	double inline GetStepSize(int h, int r, int t);

private:
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution;
	std::uniform_int_distribution<int> r_distribution;
};