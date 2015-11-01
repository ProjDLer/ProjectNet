#pragma once

#include "util.h"
#include "Dictionary.h"
#include "neuralnet.h"
#include <mutex>
#include <unordered_set>

class Reader
{
public:
	void Open(const char* input_file, Dictionary* dict);
	void Close();
	int GetSentence(int* sentence);
	long long total_words;
	static void LoadVocab();

private:

	FILE* fin;
	std::mutex reader_lock;
	char word[MAX_STRING + 1];
	Dictionary* m_dict;
	std::unordered_set<std::string> stopwords_table;
};