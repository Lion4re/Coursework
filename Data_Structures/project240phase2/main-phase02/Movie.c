/*****************************************************
 * @file   Movie.c                                   *
 * @author Paterakis Giorgos <geopat@csd.uoc.gr>     *
 *                                                   *
 * @brief Implementation for Movie.h 				 *
 * Project: Winter 2023						         *
 *****************************************************/
#include "Movie.h"

#define LOAD_FACTOR_HASH 0.45

int p;
int a;
int b;
int hashtableFlag = 0;


/**
 * @brief Find the next prime number.
 * Find the next prime number after n.
 *
 * @param n The number to start searching from
 *
 * @return The next prime number
 */
int nextPrime(int n){
	int i = 0;

	while(primes_g[i] < n){
		i++;
	}

	return primes_g[i];
}

/** 
 * @brief Initialize the hash table.
 * Initialize the hash table based on the size of the input.
 * 
 * @param size The size of the input (max_users).
 * 
 * @return void
 */
void initHashTable(int size){
	srand(time(NULL));

	p = nextPrime(size);
	a = rand() % (p-1) + 1;
	b = rand() % p;

	hashtable_size = (int)(size * LOAD_FACTOR_HASH);

	user_hashtable_p = (user_t **) malloc(sizeof(user_t *) * hashtable_size);
	if(user_hashtable_p == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		exit(EXIT_FAILURE);
	}
}

/**
 * @brief Hash function.
 * Hash function for the hash table.
 *
 * @param userID The identifier of the user
 *
 * @return The hash value
 */
int hash_function(int userID){
	return ((a * userID + b) % p) % hashtable_size;
}

/**
 * @brief Prints the list of users.
 * Prints each user's ID.
 *
 * @param head The head of the list of users.
 *
 * @return void.
 */
void printUserList(user_t *head){
	user_t *current = head;

	while(current != NULL){
		printf("\t%d\n", current->userID);
		current = current->next;
	}
	printf("\n");
}

/**
 * @brief Prints the new movies IDs in order. (New Releases Tree) 
 * Performs in-order traversal of the new releases tree and prints the movie IDs.
 *
 * @param root The root of the new releases tree.
 *
 * @return void.
 */
void inOrderPrintNewMovies(new_movie_t *root){
	if(root != NULL){
		inOrderPrintNewMovies(root->lc);
		printf("%d ", root->movieID);
		inOrderPrintNewMovies(root->rc);
	}
}

/**
 * @brief Prints the movies IDs in order. (Movies Tree)
 * Performs in-order traversal of the movies tree and prints the movie IDs.
 *
 * @param root The root of the movies tree.
 *
 * @return void.
 */
void inOrderPrintMovies(movie_t *root){
	if(root != NULL){
		inOrderPrintMovies(root->lc);
		if(root->movieID != -1){
			printf("%d ", root->movieID);
		}
		inOrderPrintMovies(root->rc);
	}
}

/**
 * @brief Prints the movies IDs of the watch histroy in order. (Watch History Tree)
 * Performs in-order traversal of the watch history tree and prints the movie IDs.
 *
 * @param root The root of the watch history tree.
 *
 * @return void.
 */
void inOrderPrintWatchHistory(userMovie_t *root){
	if(root != NULL){
		inOrderPrintWatchHistory(root->lc);
		printf("%d ", root->movieID);
		inOrderPrintWatchHistory(root->rc);
	}
}

/**
 * @brief Searches for a movie in the new releases tree.
 * Searches for a movie with a given movieID in the new releases tree.
 *
 * @param root The root of the new releases tree.
 * @param movieID The ID of the movie to search for.
 *
 * @return The node of the new movie if found, NULL otherwise.
 */
new_movie_t *newMovieSearch(new_movie_t *root, int movieID){
	if(root == NULL){
		return NULL;
	}

	if(root->movieID == movieID){
		return root;
	}

	if(root->movieID < movieID){
		return newMovieSearch(root->rc, movieID);
	}

	return newMovieSearch(root->lc, movieID);
}

/**
 * @brief Counts the number of movies in each category.
 * Performs pre-order traversal of the new releases tree.
 * For each movie, it increments the count for its category in the categoryCounts array.
 *
 * @param root The root of the new releases tree.
 * @param categoryCounts An array to store the count of movies in each category.
 *
 * @return void.
 */
void countMoviesInCategories(new_movie_t* root, int* categoryCounts) {
    if (root == NULL) {
        return;
    }

    /* Increment the count for the appropriate category */
    categoryCounts[root->category]++;

    countMoviesInCategories(root->lc, categoryCounts);
    countMoviesInCategories(root->rc, categoryCounts);
}

/**
 * @brief Fills an array with movies from a specific category.
 * Performs an in-order traversal of the new releases tree.
 * For each movie in a category, adds the movie to the array.
 *
 * @param root The root of the new releases tree.
 * @param array The array to fill with movies.
 * @param category The category of movie.
 * @param pos The current position in the array.
 *
 * @return The position in the array after filling it with movies.
 */
int fillArray(new_movie_t* root, new_movie_t** array, int category, int pos) {
    if (root == NULL) {
        return pos;
    }

    pos = fillArray(root->lc, array, category, pos);

    if (root->category == category) {
        array[pos++] = root;
    }

    pos = fillArray(root->rc, array, category, pos);

    return pos;
}

/**
 * @brief Converts a sorted array to a balanced binary search tree.
 * Takes a sorted array of movies and recursively constructs a balanced BST.
 *
 * @param array The array of movies.
 * @param start The starting index.
 * @param end The ending index.
 *
 * @return The root of the tree.
 */
movie_t *sortedArrayToBST(new_movie_t **array, int start, int end){
	if(start > end){
		return NULL;
	}

	int mid = (start + end) / 2;
	movie_t *root = (movie_t *) malloc(sizeof(movie_t));
	if(root == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		exit(EXIT_FAILURE);
	}

	root->movieID = array[mid]->movieID;
	root->year = array[mid]->year;
	root->watchedCounter = 0;
	root->sumScore = 0;
	root->lc = sortedArrayToBST(array, start, mid - 1);
	root->rc = sortedArrayToBST(array, mid + 1, end);

	return root;
}

/**
 * @brief Searches for a movie in the movie tree.
 * Searches for a movie with a given movieID in the movie tree.
 *
 * @param root The root of the movie tree.
 * @param movieID The ID of the movie.
 *
 * @return The node of the movie if found, NULL otherwise.
 */
movie_t *movieSearch(movie_t *root, int movieID){
	if(root == NULL || root->movieID == -1){
		return NULL;
	}

	if(root->movieID == movieID){
		return root;
	}

	if(root->movieID < movieID){
		return movieSearch(root->rc, movieID);
	}

	return movieSearch(root->lc, movieID);
}

/**
 * @brief Inserts a user movie into a user's watch history.
 * Inserts a new movie into a user's watch history.
 *
 * @param root The root of the user's watch history tree.
 * @param newUserMovie The new movie to insert.
 *
 * @return The root of the BST after the new movie has been inserted.
 */
userMovie_t *insertToWatchHistory(userMovie_t *root, userMovie_t *newUserMovie){
	userMovie_t *current;
	userMovie_t *parentNode;
	userMovie_t *parentValueLeaf;

	if(root == NULL){
		return newUserMovie;
	}

	current = root;
	parentNode = NULL;

	while(current != NULL){
		parentNode = current;
		if(newUserMovie->movieID < current->movieID){
			current = current->lc;
		}
		else{
			current = current->rc;
		}
	}

	/* Create the node that will hold the new 3-node tree */
	parentValueLeaf = (userMovie_t *) malloc(sizeof(userMovie_t));
	if(parentValueLeaf == NULL){
		printf("Memory not allocated. malloc() fail...\n");
		exit(EXIT_FAILURE);
	}

	/* parentValueLeaf node is actually the v', the value we want to keep in the leafs */
	parentValueLeaf->movieID	= parentNode->movieID;
	parentValueLeaf->category	= parentNode->category;
	parentValueLeaf->score		= parentNode->score;
	parentValueLeaf->parent		= parentNode;
	parentValueLeaf->lc			= NULL;
	parentValueLeaf->rc			= NULL;

	/* Parent of the newUserMovie node is the parentValueLeaf */
	newUserMovie->parent = parentNode;

    if (newUserMovie->movieID < parentNode->movieID) {
        parentNode->lc = newUserMovie;
		parentNode->rc = parentValueLeaf;
    } else {
        parentNode->lc = parentValueLeaf;
		parentNode->rc = newUserMovie;
    }

    return root;
}


/**
 * @brief Prints the movie IDs of the leaf nodes of a leaf-oriented BST in order.
 * Performs an in-order traversal of the leaf-oriented BST.
 * For each leaf node, it prints the movie ID.
 *
 * @param node The root of the leaf-oriented BST.
 *
 * @return void.
 */
void printLeafNodesInOrder(userMovie_t *node) {
    if (node == NULL) {
        return;
    }
    printLeafNodesInOrder(node->lc);

    if (node->lc == NULL && node->rc == NULL) {
        printf("%d ", node->movieID);
    }

    printLeafNodesInOrder(node->rc);
}

/**
 * @brief Deletes a user's watch history.
 * This function recursively deletes all nodes in user's watch history tree.
 *
 * @param root A pointer to the root of the watch history tree.
 *
 * @return void.
 */
void deleteWatchHistory(userMovie_t **root){
	if(*root == NULL){
		return;
	}

	deleteWatchHistory(&((*root)->lc));
	deleteWatchHistory(&((*root)->rc));

	free(*root);
	*root = NULL;
}

/**
 * @brief Calculates the average score of a movie.
 * Computes the average score of a movie.
 *
 * @param movie The movie to calculate the average score.
 *
 * @return The average score of the movie.
 */
double calculateAverageScore(movie_t *movie){
    if(movie->watchedCounter == 0){ 
		return 0;
	}
    return (double) movie->sumScore / movie->watchedCounter;
}

/**
 * @brief Counts the movies with an average score greater than a given score.
 * Traverses the movie tree and increments the count if a movie's average score is greater than the score.
 *
 * @param node The root of the movie tree.
 * @param score The score to compare.
 * @param count A pointer to the count of movies.
 *
 * @return void.
 */
void countMovies(movie_t *node, int score, int *count) {
    if (node == NULL || node->movieID == -1) {
        return;
    }

    countMovies(node->lc, score, count);

    if(calculateAverageScore(node) > score && node->watchedCounter != 0){
        (*count)++;
    }

    countMovies(node->rc, score, count);
}

/**
 * @brief Traverses the movie tree and adds movies with an average score greater than a given score to an array.
 * Traverses the movie tree in-order and adds each movie whose average score is greater than score to the array.
 *
 * @param root The root of the movie tree.
 * @param score The score to compare.
 * @param array The array to fill.
 * @param pos A pointer to the current position in the array.
 *
 * @return void.
 */
void traverseAndPoint(movie_t *root, int score, movie_t **array, int *pos){
	if (root == NULL) {
		return;
	}

	traverseAndPoint(root->lc, score, array, pos);

	if(calculateAverageScore(root) > score){
		array[*pos] = root;
		(*pos)++;
	}

	traverseAndPoint(root->rc, score, array, pos);
}

/**
 * @brief Rearranges a subtree to maintain the heap property.
 *
 * @param array The heap.
 * @param n The number of elements in the heap.
 * @param i The index of the current node.
 *
 * @return void.
 */
void heapify(movie_t **array, int n, int i){
	movie_t *swap;
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if(left < n && calculateAverageScore(array[left]) > calculateAverageScore(array[largest])){
        largest = left;
    }

    if(right < n && calculateAverageScore(array[right]) > calculateAverageScore(array[largest])){
        largest = right;
    }

    if(largest != i){
        swap = array[i];
        array[i] = array[largest];
        array[largest] = swap;

        heapify(array, n, largest);
    }
}

/**
 * @brief Sorts an array of movies using heapsort.
 *
 * @param array The array of movies to sort.
 * @param n The number of movies in the array.
 *
 * @return void.
 */
void heapsort(movie_t **array, int n){
	int i;
	movie_t *swap;

    for(i = n / 2 - 1; i >= 0; i--){
        heapify(array, n, i);
    }

    for(i = n - 1; i > 0; i--){
        swap = array[0];
        array[0] = array[i];
        array[i] = swap;

        heapify(array, i, 0);
    }
}

/**
 * @brief Finds the next leaf node in in-order traversal.
 * Starts from a  node and finds the next leaf node.
 *
 * @param node The starting node.
 *
 * @return The next leaf node in in-order traversal, or NULL if there is no next leaf node.
 */
userMovie_t *findNextLeaf(userMovie_t *node){
	userMovie_t *nextLeaf;
	userMovie_t *parentNode;

	parentNode = node->parent;
	while(parentNode != NULL && parentNode->rc == node){
		node = parentNode;
		parentNode = parentNode->parent;
	}

	if(parentNode == NULL){
		return NULL;
	}

	nextLeaf = parentNode->rc;
	while(nextLeaf->lc != NULL){
		nextLeaf = nextLeaf->lc;
	}

	return nextLeaf;
}

/**
 * @brief Deletes the new releases tree.
 * Deletes all nodes in the new releases tree.
 *
 * @param root A pointer to the root of the new releases tree.
 *
 * @return void.
 */
void deleteNewReleasesTree(new_movie_t **root){
	if(*root == NULL){
		return;
	}

	deleteNewReleasesTree(&((*root)->lc));
	deleteNewReleasesTree(&((*root)->rc));

	free(*root);
	*root = NULL;
}

/**
 * @brief Deletes the movie tree.
 * Deletes all nodes in the movie tree.
 *
 * @param root A pointer to the root of the movie tree.
 *
 * @return void.
 */
void deleteMovieTree(movie_t *root){
	if(root == NULL){
		return;
	}

	deleteMovieTree(root->lc);
	deleteMovieTree(root->rc);

	free(root);
	root = NULL;
}

/**
 * @brief Returns the height of a node in the new releases tree.
 * Returns the height of a given node.
 *
 * @param node A pointer to the node in the new releases tree.
 *
 * @return The height of the node, or 0 if the node is NULL.
 */
int height(new_movie_t *node){
    if (node == NULL){
        return 0;
	}

    return node->height; 
}

/**
 * @brief Returns the maximum of two integers.
 * Compares two integers and returns the larger one.
 *
 * @param a The first integer.
 * @param b The second integer.
 *
 * @return The larger of the two integers.
 */
int maximum(int a, int b){ 
    if (a > b){
        return a;
	}
    else{
        return b;
	}
}

/**
 * @brief Gets the balance factor of a node in the new releases tree.
 * The balance factor is the difference between the heights of the left child and the right child.
 *
 * @param node The node to calculate the balance factor.
 *
 * @return The balance factor of the node. 
 *         If the node is NULL, the function returns 0.
 */
int getBalance(new_movie_t *node){ 
    if (node == NULL){
        return 0;
	}

    return height(node->lc) - height(node->rc); 
}

/**
 * @brief Creates a new movie node.
 * Allocates memory for a new movie node and initializes its fields.
 *
 * @param movieID The ID of the movie.
 * @param category The category of the movie.
 * @param year The release year of the movie.
 *
 * @return A pointer to the created new movie node.
 */
new_movie_t* createNewMovieNode(int movieID, int category, int year){ 
    new_movie_t* newMovie;

	newMovie = (new_movie_t*) malloc(sizeof(new_movie_t));
	if(newMovie == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		exit(EXIT_FAILURE);
	}

	newMovie->movieID = movieID; 
	newMovie->category = category;
	newMovie->year = year;
	newMovie->lc = NULL; 
	newMovie->rc = NULL; 
	newMovie->height = 1;
	return(newMovie);
}

/**
 * @brief Right rotation on a node in the AVL new releases tree.
 * Right rotation on a node to keep the balance in the AVL new releases tree.
 *
 * @param y The node to perform the right rotation.
 *
 * @return The new root of the subtree after the rotation.
 */
new_movie_t *rightRotate(new_movie_t *y){
	new_movie_t *x;
	new_movie_t *T2;

    x = y->lc;
    T2 = x->rc;
    x->rc = y; 
    y->lc = T2; 

    y->height = maximum(height(y->lc), height(y->rc)) + 1; 
    x->height = maximum(height(x->lc), height(x->rc)) + 1; 
    return x; 
} 

/**
 * @brief Left rotation on a node in the AVL new releases tree.
 * Left rotation on a node to keep the balance in the AVL new releases tree.
 *
 * @param x The node to perform the left rotation.
 *
 * @return The new root of the subtree after the rotation.
 */
new_movie_t *leftRotate(new_movie_t *x){
	new_movie_t *y;
	new_movie_t *T2;

    y = x->rc; 
    T2 = y->lc;

    y->lc = x; 
    x->rc = T2; 

    x->height = maximum(height(x->lc), height(x->rc)) + 1; 
    y->height = maximum(height(y->lc), height(y->rc)) + 1; 

    return y;
} 

/**
 * @brief Inserts a new movie into the new releases tree.
 * Inserts a new movie with a given movieID, category, and year into the movie tree.
 * The function also balances the tree after insertion if necessary.
 * 
 * @param root The root of the movie tree.
 * @param movieID The ID of the new movie.
 * @param category The category of the new movie.
 * @param year The year of the new movie.
 *
 * @return The root of the movie tree after the insertion.
 */
new_movie_t* insertAVL(new_movie_t* root, int movieID, int category, int year){ 
    if (root == NULL){ 
        return(createNewMovieNode(movieID, category, year));
	}

    if(movieID < root->movieID){
        root->lc  = insertAVL(root->lc, movieID, category, year);
	}
    else if(movieID > root->movieID){
        root->rc = insertAVL(root->rc, movieID, category, year);
	}
    else{
        return root;
	}

    root->height = 1 + maximum(height(root->lc), height(root->rc)); 

    int balance = getBalance(root);

	/* If this node becomes unbalanced, then there are 4 cases */
    /* Left Left Case */
    if (balance > 1 && movieID < root->lc->movieID){
        return rightRotate(root); 
	}
	/*Right Right Case */
    if (balance < -1 && movieID > root->rc->movieID){
        return leftRotate(root);
	}
	/* Left Right Case */
    if (balance > 1 && movieID > root->lc->movieID){ 
        root->lc =  leftRotate(root->lc);
        return rightRotate(root);
    }
	/* Right Left Case */
    if (balance < -1 && movieID < root->rc->movieID){ 
        root->rc = rightRotate(root->rc); 
        return leftRotate(root);
    }

    return root; 
}


/**
 * @brief Creates a new user.
 * Creates a new user with userID as its identification.
 *
 * @param userID The new user's identification
 *
 * @return 1 on success
 *         0 on failure
 */

int register_user(int userID){
	user_t *current;
	int hashValue;

	hashValue = hash_function(userID);

	/* Checking that the user doesn't already exist */
	current = user_hashtable_p[hashValue];
	while(current != NULL){
		if(current->userID == userID){
			printf("\nR %d\n", userID);
			printf("\t User %d already exists\n", userID);
			return 0;
		}
		current = current->next;
	}

	user_t *newUser = (user_t *) malloc(sizeof(user_t));
	if(newUser == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		return 0;
	}

	newUser->userID = userID;
	newUser->history = NULL;
	newUser->next = NULL;

	if(user_hashtable_p[hashValue] == NULL){
		user_hashtable_p[hashValue] = newUser;
	}
	else{
		current = user_hashtable_p[hashValue];
		while(current->next != NULL){
			current = current->next;
		}
		current->next = newUser;
	}

	printf("\nR %d\n", userID);
	printf("Chain %d of Users:\n", hashValue);
	printUserList(user_hashtable_p[hashValue]);
	printf("DONE\n");

	return 1;
}

/**
 * @brief Deletes a user.
 * Deletes a user with userID from the system, along with users' history tree.
 *
 * @param userID The new user's identification
 *
 * @return 1 on success
 *         0 on failure
 */

int unregister_user(int userID){
	user_t *current;
	user_t *previous = NULL;
	int flag = 0;

	int hashValue = hash_function(userID);

	current = user_hashtable_p[hashValue];

	/* Traverse the chain to find the user */
	while(current != NULL){
		if(current->userID == userID){
			flag = 1;	/* Indicates that user found */
			break;
		}
		previous = current;
		current = current->next;
	}

	if(flag == 0){
		printf("\nU %d\n", userID);
		printf("\t User %d not found\n", userID);
		return 0;
	}

	/* Delete the user's watch history */
	if(current->history != NULL){
		deleteWatchHistory(&(current->history));
		current->history = NULL;
	}

    if (previous == NULL) {
        user_hashtable_p[hashValue] = current->next;
    } else {
        previous->next = current->next;
    }

    free(current);
	current = NULL;

	printf("\nU %d\n", userID);
	printf("Chain %d of Users:\n", hashValue);
	printUserList(user_hashtable_p[hashValue]);
	printf("DONE\n");

	return 1;
}

/**
 * @brief Add new movie to new release binary tree.
 * Create a node movie and insert it in 'new release' binary tree.
 *
 * @param movieID The new movie identifier
 * @param category The category of the movie
 * @param year The year movie released
 *
 * @return 1 on success
 *         0 on failure
 */

int add_new_movie(int movieID, int category, int year){
	new_movie_t *current;
	
	/* Check if movie already exists */
	current = newMovieSearch(new_releases, movieID);
	if(current != NULL){
		printf("\nA %d %d %d\n", movieID, category, year);
		printf("\t Movie %d already exists\n", movieID);
		return 0;
	}

	new_releases = insertAVL(new_releases, movieID, category, year);

	printf("\nA %d %d %d\n", movieID, category, year);
	printf("New releases Tree:\n\t");
	inOrderPrintNewMovies(new_releases);
	printf("\nDONE\n");

	return 1;
}

/**
 * @brief Distribute the movies from new release binary tree to the array of categories.
 *
 * @return 1 on success
 *         0 on failure
 */

int distribute_movies(void){
	int categoryCounts[6] = {0, 0, 0, 0, 0, 0};
	int i;
	new_movie_t **tempCategoryTables;

	/* Count the movies in each category */
	countMoviesInCategories(new_releases, categoryCounts);

	for(i = 0; i < 6; i++){
		if(categoryCounts[i] == 0){
			continue;
		}

		/* Allocate memory for the temporary table */
		tempCategoryTables = (new_movie_t **) malloc(sizeof(new_movie_t *) * categoryCounts[i]);
		if(tempCategoryTables == NULL){
			fprintf(stderr, "Memory not allocated. malloc() fail...\n");
			return 0;
		}

		/* Fill the temporary table */
		fillArray(new_releases, tempCategoryTables, i, 0);

		/* Create the BST from the temporary array */
		categoryArray[i]->movie = sortedArrayToBST(tempCategoryTables, 0, categoryCounts[i] - 1);

		/* Free the memory of the temporary table */
		free(tempCategoryTables);
		tempCategoryTables = NULL;
	}
	deleteNewReleasesTree(&new_releases);

	printf("\nD\n");
	printf("Movie Category Array:\n\t");
	for(i = 0; i < 6; i++){
		printf("Category %d:\n\t\t", i);
		inOrderPrintMovies(categoryArray[i]->movie);
		printf("\n\t");
	}
	printf("\nDONE\n");

	return 1;
}

/**
 * @brief User rates the movie with identification movieID with score
 *
 * @param userID The identifier of the user
 * @param category The Category of the movie
 * @param movieID The identifier of the movie
 * @param score The score that user rates the movie with id movieID
 *
 * @return 1 on success
 *         0 on failure
 */

int watch_movie(int userID, int category, int movieID, int score){
	movie_t *watched_movie;
	userMovie_t *newUserMovie;
	user_t *currentUser;
	int hashValue;

	if(category < 0 || category > 5){
		printf("\nW %d %d %d %d\n", userID, category, movieID, score);
		printf("\t Category %d, does not exist", category);
		printf("\nDONE\n");
		return 0;
	}

	watched_movie = movieSearch(categoryArray[category]->movie, movieID);

	if(watched_movie == NULL){
		printf("\nW %d %d %d %d\n", userID, category, movieID, score);
		printf("\t Movie %d in category %d, does not exist", movieID, category);
		printf("\nDONE\n");
		return 0;
	}
	else{
		watched_movie->watchedCounter++;
		watched_movie->sumScore += score;
	}

	hashValue = hash_function(userID);

	/* Checking that the user exists */
	currentUser = user_hashtable_p[hashValue];
	while(currentUser != NULL){
		if(currentUser->userID == userID){
			break;
		}
		currentUser = currentUser->next;
	}

	/* User does not exist error message */
	if(currentUser == NULL){
		printf("\nW %d %d %d %d\n", userID, category, movieID, score);
		printf("\t User %d does not exist", userID);
		printf("\nDONE\n");
		return 0;
	}

	newUserMovie = (userMovie_t *) malloc(sizeof(userMovie_t));
	if(newUserMovie == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		return 0;
	}

	newUserMovie->movieID = movieID;
	newUserMovie->category = category;
	newUserMovie->score = score;
	newUserMovie->parent = NULL;
	newUserMovie->lc = NULL;
	newUserMovie->rc = NULL;

	currentUser->history = insertToWatchHistory(currentUser->history, newUserMovie);

	printf("\nW %d %d %d %d\n", userID, category, movieID, score);
	printf("History Tree of user %d:\n\t", userID);
	printLeafNodesInOrder(currentUser->history);
	printf("\nDONE\n");
	
	return 1;
}

/**
 * @brief Identify the best rating score movie and cluster all the movies of a category.
 *
 * @param userID The identifier of the user
 * @param score The minimum score of a movie
 *
 * @return 1 on success
 *         0 on failure
 */

int filter_movies(int userID, int score){
	int hashValue;
	int count;
	int i;
	int index;
	user_t *currentUser;
	movie_t **tempTable;
	movie_t *tempMovie;

	hashValue = hash_function(userID);

	/* Checking that the user exists */
	currentUser = user_hashtable_p[hashValue];
	while(currentUser != NULL){
		if(currentUser->userID == userID){
			break;
		}
		currentUser = currentUser->next;
	}

	/* User does not exist error message */
	if(currentUser == NULL){
		printf("\nF %d %d\n", userID, score);
		printf("\t User %d does not exist", userID);
		printf("\nDONE\n");
		return 0;
	}

	count = 0;

	for(i = 0; i < 6; i++){
		tempMovie = categoryArray[i]->movie;
		countMovies(tempMovie, score, &count);
	}

	if(count == 0){
		printf("\nF %d %d\n", userID, score);
		printf("\t No movies found");
		printf("\nDONE\n");
		return 0;
	}

	tempTable = (movie_t **) malloc(sizeof(movie_t *) * count);
	if(tempTable == NULL){
		fprintf(stderr, "Memory not allocated. malloc() fail...\n");
		return 0;
	}

	index = 0;

	for(i = 0; i < 6; i++){
		tempMovie = categoryArray[i]->movie;
		traverseAndPoint(tempMovie, score, tempTable, &index);
	}

	heapsort(tempTable, count);

	printf("\nF %d %d\n\t", userID, score);
	printf("Movie: \t Score:\n");
	for(i = 0; i < count; i++){
		printf("\t%d \t %.2f\n", tempTable[i]->movieID, calculateAverageScore(tempTable[i]));
	}
	printf("\nDONE\n");

	
	free(tempTable);
	tempTable = NULL;

	return 1;
}

/**
 * @brief Find movies from categories withn median_score >= score t
 *
 * @param userID The identifier of the user
 * @param category Array with the categories to search.
 * @param score The minimum score the movies we want to have
 *
 * @return 1 on success
 *         0 on failure
 */

int user_stats(int userID){
	int hashValue;
	int scoreSum;
	int counter;
	double averageRate;
	user_t *currentUser;
	userMovie_t *leftMostLeaf;

	hashValue = hash_function(userID);

	/* Checking that the user exists */
	currentUser = user_hashtable_p[hashValue];
	while(currentUser != NULL){
		if(currentUser->userID == userID){
			break;
		}
		currentUser = currentUser->next;
	}

	/* User does not exist error message */
	if(currentUser == NULL){
		printf("\nQ %d\n", userID);
		printf("\t User %d does not exist", userID);
		printf("\nDONE\n");
		return 0;
	}

	/* User has no history error */
	if(currentUser->history == NULL){
		printf("\nQ %d\n", userID);
		printf("\t User %d has no history", userID);
		printf("\nDONE\n");
		return 0;
	}

	scoreSum = 0;
	counter = 0;

	leftMostLeaf = currentUser->history;
	while(leftMostLeaf->lc != NULL){
		leftMostLeaf = leftMostLeaf->lc;
	}

	while(leftMostLeaf != NULL){
		scoreSum += leftMostLeaf->score;
		counter++;
		leftMostLeaf = findNextLeaf(leftMostLeaf);
	}

	averageRate = (double) scoreSum / counter;

	printf("\nQ %d %.2f", userID, averageRate);
	printf("\nDONE\n");

	return 1;
}

/**
 * @brief Search for a movie with identification movieID in a specific category.
 *
 * @param movieID The identifier of the movie
 * @param category The category of the movie
 *
 * @return 1 on success
 *         0 on failure
 */

int search_movie(int movieID, int category){
	
	if(category < 0 || category > 5){
		printf("\nI %d %d\n", movieID, category);
		printf("\t Category %d, does not exist", category);
		printf("\nDONE\n");
		return 0;
	}

	if(movieSearch(categoryArray[category]->movie, movieID) == NULL){
		printf("\nI %d %d\n", movieID, category);
		printf("\t Movie %d in category %d, does not exist", movieID, category);
		printf("\nDONE\n");
		return 0;
	}
	else{
		printf("\nI %d %d\n", movieID, category);
		printf("\t Movie %d in category %d, exists", movieID, category);
		printf("\nDONE\n");
	}

	return 1;
}

/**
 * @brief Prints the movies in movies categories array.
 * @return 1 on success
 *         0 on failure
 */

int print_movies(void){
	int i;

	if (categoryArray == NULL) {
        printf("Error: categoryArray is NULL\n");
        return 0;
    }

	printf("\nM\n");
	printf("Movie Category Array:\n\t");
	for(i = 0; i < 6; i++){
		printf("Category %d:\n\t\t", i);
		inOrderPrintMovies(categoryArray[i]->movie);
		printf("\n\t");
	}
	printf("\nDONE\n");

	return 1;
}

/**
 * @brief Prints the users hashtable.
 * @return 1 on success
 *         0 on failure
 */

int print_users(void){
	int i;
	user_t *current;

	if(user_hashtable_p == NULL){
		printf("Error: user_hashtable_p is NULL\n");
		return 0;
	}

	printf("\nP\n");
	for(i = 0; i < hashtable_size; i++){
		printf("Chain %d of Users:\n", i);
		
		current = user_hashtable_p[i];
		while(current != NULL){
			printf(" \n");
			printf("\tUser: %d\n", current->userID);
			printf("\tHistory Tree:\n\t");
			inOrderPrintWatchHistory(current->history);
			current = current->next;
		}
		printf("\n");
	}
	printf("DONE\n");

	return 1;
}
