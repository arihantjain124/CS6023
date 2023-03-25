#include <stdio.h>
#include <cuda.h>

#define N 20
struct node {
	struct node *next;
	int data;
};

struct node *createNode(int ii) {
	struct node *nn = (struct node *)malloc(sizeof(struct node));
	nn->data = ii;
	nn->next = NULL;

	return nn;
}
struct node *createList() {
	struct node *head = NULL;

	for (int ii = 20; ii > 0; --ii) {
		struct node *nn = createNode(ii);
		nn->next = head;
		head = nn;
	}
	return head;
}
__device__ __host__ void printList(struct node *head) {
	if (head) {
		printf("%d ", head->data);
		printList(head->next);
	} else {
		printf("\n");
	}
}
__global__ void printListGPU(struct node *head) {
	printList(head);
}
struct node *copyNode(struct node *nn) {
	struct node *nngpu;
	cudaMalloc(&nngpu, sizeof(struct node));
	cudaMemcpy(nngpu, nn, sizeof(struct node), cudaMemcpyHostToDevice);
	return nngpu;
}
struct node *copyList(struct node *head) {
	if (!head) return NULL;

	struct node nn;
	nn.next = copyList(head->next);
	nn.data = head->data;
	return copyNode(&nn);
}
int main() {
	struct node *head = createList();
	struct node *gpuhead = copyList(head);

	printList(head);
	printListGPU<<<1, 1>>>(gpuhead);
	cudaDeviceSynchronize();

	return 0;
}
