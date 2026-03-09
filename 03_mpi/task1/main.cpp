#include <iostream>
#include <string>
#include <vector>

#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>

#include "mpi.h"

#define TAG_START_DATA 1
#define TAG_RESULT_DATA 2

MPI_Status status;

using boost::uuids::detail::md5;
using namespace std;

char alphabet[] = { "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" };
int alphabet_length = sizeof(alphabet) / sizeof(alphabet[0]) - 1;

const int password_length = 3;

string generatePassword(int length);
string digestToString(const md5::digest_type& digest);
string bruteForce(int length, int start, int end, const md5::digest_type& target_digest);
bool digestsEqual(const md5::digest_type& a, const md5::digest_type& b);

int main(int argc, char** argv) {
	srand(time(NULL));

	MPI_Init(&argc, &argv);

	int thread_id, threads, slaves;
	double t_start, t_end;

	MPI_Comm_rank(MPI_COMM_WORLD, &thread_id);
	MPI_Comm_size(MPI_COMM_WORLD, &threads);

	t_start = MPI_Wtime();
	if (thread_id == 0) {

		slaves = threads - 1;

		string random_password = generatePassword(password_length);

		md5 pass_hash;
		md5::digest_type pass_digest;
		pass_hash.process_bytes(random_password.data(), random_password.size());
		pass_hash.get_digest(pass_digest);

		cout << "Generated password: " << random_password << "\n";
		cout << "MD5: " << digestToString(pass_digest) << "\n";

		int iter_count = (int)pow(alphabet_length, password_length);
		int iter_piece = iter_count / slaves;
		int offset = 0;

		for (int dest = 1; dest <= slaves; dest++) {
			if (dest == slaves) {
				iter_piece += iter_count % slaves + 1;
			}

			int start = offset;
			int end = offset + iter_piece - 1;

			MPI_Send(&start, 1, MPI_INT, dest, TAG_START_DATA, MPI_COMM_WORLD);
			MPI_Send(&end, 1, MPI_INT, dest, TAG_START_DATA, MPI_COMM_WORLD);
			MPI_Send(&pass_digest, 4, MPI_UINT32_T, dest, TAG_START_DATA, MPI_COMM_WORLD);

			offset += iter_piece;
		}

		for (int src = 1; src <= slaves; src++) {
			string brut_password;
			int char_count;
			MPI_Probe(src, TAG_RESULT_DATA, MPI_COMM_WORLD, &status);
			MPI_Get_count(&status, MPI_CHAR, &char_count);
			for (int i = 0; i < char_count; i++) {
				brut_password.push_back(0);
			}
			MPI_Recv(&brut_password[0], char_count, MPI_CHAR, src, TAG_RESULT_DATA, MPI_COMM_WORLD, &status);
			cout << "Thread " << src << ", result: " << brut_password << "\n";
		}
	}

	if (thread_id > 0) {
		int start, end;
		md5::digest_type target_digest;

		MPI_Recv(&start, 1, MPI_INT, 0, TAG_START_DATA, MPI_COMM_WORLD, &status);
		MPI_Recv(&end, 1, MPI_INT, 0, TAG_START_DATA, MPI_COMM_WORLD, &status);
		MPI_Recv(&target_digest, 4, MPI_UINT32_T, 0, TAG_START_DATA, MPI_COMM_WORLD, &status);

		string result = bruteForce(password_length, start, end, target_digest);

		MPI_Send(result.c_str(), result.size(), MPI_CHAR, 0, TAG_RESULT_DATA, MPI_COMM_WORLD);
	}

	t_end = MPI_Wtime();

	if (thread_id == 0) {
		cout << "Execution time: " << t_end - t_start << " seconds\n";
	}

	MPI_Finalize();

	return 0;
}

string generatePassword(int length) {
	string password;
	for (int i = 0; i < length; i++) {
		int ascii = 0;
		switch (rand() % 3) {
		case(0):
			ascii = 48 + (rand() % 10);
			break;
		case(1):
			ascii = 65 + (rand() % 26);
			break;
		case(2):
			ascii = 97 + (rand() % 26);
			break;
		}
		password += char(ascii);
	}
	return password;
}

string digestToString(const md5::digest_type& digest) {
	const auto charDigest = reinterpret_cast<const char*>(&digest);
	string result;
	boost::algorithm::hex(charDigest, charDigest + sizeof(md5::digest_type), back_inserter(result));
	return result;
}

bool digestsEqual(const md5::digest_type& a, const md5::digest_type& b) {
	return memcmp(&a, &b, sizeof(md5::digest_type)) == 0;
}

string bruteForce(int length, int start, int end, const md5::digest_type& target_digest) {
	string brut_password(length, '0');

	for (int i = start; i <= end; i++) {
		int cur_iter = i;
		for (int j = 0; j < length; j++) {
			int divider = (int)pow(alphabet_length, length - (j + 1));
			int cur_pos = cur_iter / divider;
			brut_password[j] = alphabet[cur_pos];
			cur_iter -= divider * cur_pos;
		}

		md5 brut_hash;
		md5::digest_type brut_digest;
		brut_hash.process_bytes(brut_password.data(), brut_password.size());
		brut_hash.get_digest(brut_digest);

		if (digestsEqual(brut_digest, target_digest)) {
			cout << "\nPassword found: " << brut_password << "\n\n";
			return brut_password;
		}
	}

	return string(length, '-');
}
