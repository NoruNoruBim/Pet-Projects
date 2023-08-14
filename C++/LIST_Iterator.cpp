#include <iostream>
#include <string>
#include <memory>

using namespace std;


class Node {
    public:
        using node_ptr = std::unique_ptr<Node>;
        string value;
        node_ptr next;
        
        Node() {
            this->value = "";
            this->next = nullptr;
        }
        
        Node(string s) {
            this->value = s;
            this->next = nullptr;
        }
        
        ~Node() {}
};


class Iterator {

    private:
        const Node *current_node = nullptr;
    
    public:
        using node_ptr = std::unique_ptr<Node>;
        Iterator() noexcept : current_node(nullptr) {};
        Iterator(const node_ptr &node) noexcept : current_node(node.get()){};

        Iterator& operator++() noexcept
        {
            if (current_node != nullptr)
            {
                if (current_node->next == nullptr) {
                    current_node = nullptr;
                } else if (current_node->next->next == nullptr) {
                    current_node = nullptr;
                } else {
                    current_node = current_node->next->next.get();
                }
            }
            return *this;
        };

        Iterator operator++(int) noexcept
        {
            Iterator tempIter = *this;
            ++*this;
            return tempIter;
        };

        bool operator!=(const Iterator &other) const noexcept
        {
            return this->current_node != other.current_node;
        };

        string operator*() const noexcept
        {
            return this->current_node->value;
        };
};

class LinkedList {
    
    private:
        using node_ptr = std::unique_ptr<Node>;
        node_ptr root;
    
    public:
        friend class Iterator;
    
        LinkedList() {
            this->root = nullptr;
        }
        
        LinkedList(string s) {
            this->root = make_unique<Node>(s);
        }
        
        void push_back(string s) {
            if (this->root == nullptr) {
                this->root = make_unique<Node>(s);
                return;
            }
            
            Node* tmp = this->root.get();
            while (tmp->next != nullptr) {
                tmp = tmp->next.get();
            }
            
            tmp->next = make_unique<Node>(s);
        }
        
        void printList() {
            Node* tmp = this->root.get();
            while (tmp != nullptr) {
                cout << tmp->value << " ";
                tmp = tmp->next.get();
            }
            cout << endl;
        }
        
        Iterator begin() const noexcept {
            return Iterator(this->root);
        }

        Iterator end() const noexcept {
            return Iterator();
        }
        
        char& operator[](int ind) {
            Node* tmp = this->root.get();
            int sum = tmp->value.size();
            while (ind >= sum) {
                tmp = tmp->next.get();
                sum += tmp->value.size();
            }
            
            return tmp->value[ind - sum + tmp->value.size()];
        }
        
        ~LinkedList() {}
};


int main() {

    LinkedList lst;
    lst.push_back("qwert");
    lst.push_back("asdf");
    lst.push_back("zxcv");
    lst.push_back("bhjk");
    lst.push_back("aaaaaa");

    lst.printList();

    for (auto i = lst.begin(); i != lst.end(); i++) {
        cout << *i << " ";
    } cout << endl;

    cout << lst[3] << endl;
    cout << lst[6] << endl;
    cout << lst[12] << endl;
    cout << lst[13] << endl;
    cout << lst[16] << endl;
    
    return 0;
}
