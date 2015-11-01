#include "reader.h"
#include "HuffmanEncoder.h"

extern Dictionary* g_dict;
extern HuffmanEncoder g_encoder;
extern Reader g_reader;

void Reader::Open(const char* input_file, Dictionary* dict)
{
	
	m_dict = dict;
	fin = fopen(input_file, "r");
	stopwords_table.clear();
	if(Opt::stopwords)
	{
		FILE* fid = fopen(Opt::sw_file, "r");
		while (Util::ReadWord(word, fid))
		{
			stopwords_table.insert(word);
		}
		fclose(fid);
	}

	reader_lock.lock();
	if (total_words == 0)
	{
		FILE* fid = fopen(Opt::train_file, "r");
		while (Util::ReadWord(word, fid))
		{
			total_words++;
		}
		fclose(fid);
	}
	reader_lock.unlock();
}

void Reader::Close()
{
	fclose(fin);
	stopwords_table.clear();
}

int Reader::GetSentence(int* sentence)
{
	reader_lock.lock();
	int length = 0, word_idx;
	int idx_end_of_sentence = m_dict->GetWordIdx("</s>");
	while (1)
	{
		if (!Util::ReadWord(word, fin))
			break;
		word_idx = m_dict->GetWordIdx(word);
		if (word_idx == -1)
			continue;
		if (word_idx == idx_end_of_sentence)
		{
			NN::train_count += 2 * Opt::window_size;
			if (length == 0)
				continue;
			else break;
		}
		if (Opt::stopwords && stopwords_table.count(word))
		{
			NN::train_count += 2 * Opt::window_size;
			continue;
		}
		if (Opt::sample > 0 && !Sampler::WordSampling(m_dict->GetWordInfo(word_idx)->freq, total_words))
		{
			NN::train_count += 2 * Opt::window_size;
			continue;
		}
		sentence[length++] = word_idx;
		if (length >= MAX_SENTENCE_LENGTH)
			break;
		if (NN::progress >= 5)
		{
			length = 0;
			break;
		}
	}

	reader_lock.unlock();
	return length;
}

void Reader::LoadVocab()
{
	printf("Loading Vocabulary...\n");
	char word[MAX_STRING];
	FILE* fid;
	g_reader.total_words = 0;
	if (Opt::read_vocab_file)
	{
		fid = fopen(Opt::read_vocab_file, "r");
		int word_freq;
		while (fscanf(fid, "%s %d", word, &word_freq) != EOF)
		{
			g_dict->Insert(word, word_freq);
		}
	} else
	{
		fid = fopen(Opt::train_file, "r");
		long long cnt = 0;
		while (Util::ReadWord(word, fid))
		{
			if ((++cnt) % 100000 == 0)
				printf("%lldK%c", cnt / 1000, 13);
			g_dict->Insert(word);
		}
	}
	g_dict->RemoveWordsLessThan(Opt::min_count);
	printf("Dictionary size: %d\n", g_dict->Size());

	for (int i = 0; i < g_dict->Size(); ++i)
		g_reader.total_words += g_dict->GetWordInfo(i)->freq;
	printf("Words in training file: %I64d\n", g_reader.total_words);
	if (Opt::hs)
		g_encoder.BuildFromTermFrequency(g_dict);
	fclose(fid);
}