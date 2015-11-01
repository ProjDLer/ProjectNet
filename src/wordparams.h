#pragma once

#include "util.h"

class WordParams
{
public:
	static int* relat_cnt;
	static real** p_embedding;
	static real** p_relat_prior;
	static int** p_relats;
	static void InitRelatCnt();
	static void InitWordParams();
};
