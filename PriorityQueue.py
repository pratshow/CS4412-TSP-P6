# Brandon Watkins
# CS 4412

import math


class PrioritizedNode:

    def __init__(self, element, priority = math.inf):
        self.priority = priority
        self.element = element

    def delete(self):
        element = self.element
        self.__delattr__("priority")
        self.__delattr__("element")
        return element


class PrioritizedHeapNode(PrioritizedNode):

    def __init__(self, element, priority = math.inf):
        super().__init__(element, priority)
        self.parent = self
        self.children = []

    # For garbage collection
    def delete(self):
        element = self.element
        self.__delattr__("parent")
        self.__delattr__("priority")
        self.__delattr__("children")
        self.__delattr__("element")
        return element

    def __hash__(self):
        try:
            return hash(self.element)
        except:
            if str(self.element.toString()):
                return hash(str(self.element.toString()))


class PriorityQueue:

    def __init__(self):
        self.queue = []

    def makeQueue(self, elements, defaultPriority):
        pass

    def insert(self, element, priority):
        pass

    def deleteMin(self):
        pass

    def decreaseKey(self, element):
        pass

    def changePriority(self, element, priority):
        pass

    def notEmpty(self):
        return self.queue is not None and len(self.queue) > 0


class DAryHeapPriorityQueue(PriorityQueue):

    def __init__(self, d = 3, elementList = [], queueSizeLimit = -1):
        super().__init__()
        # d is the number of children a node can have.
        self.d = d
        self.dic = dict()
        self.queue = []
        self.queueSizeLimit = queueSizeLimit
        if len(elementList) > 0:
            self.makeQueue(elementList)

    # Creates a priority queue with all priority levels the same
    def makeQueue(self, elements, defaultPriority = math.inf):
        for element in elements:
            if self.queueSizeLimit > 0 and len(self.queue) >= self.queueSizeLimit: break
            newNode = PrioritizedHeapNode(element, defaultPriority)
            if len(self.queue) > 0:
                newNode.parent = self.queue[math.ceil(len(self.queue) / self.d) - 1]
                newNode.parent.children.append(newNode)
            self.queue.append(newNode)
            self.dic[hash(newNode)] = len(self.queue) - 1
        return self

    # Insert a new node into the priority queue
    def insert(self, element, priority = math.inf):
        # Create a new node
        newNode = PrioritizedHeapNode(element, priority)

        # Add parent node
        if len(self.queue) > 0:
            newNode.parent = self.queue[math.floor(len(self.queue) / self.d)]
            newNode.parent.children.append(newNode)

        # Place node at the end of the queue.
        self.queue.append(newNode)

        # Place the node pointer in the dictionary
        self.dic[hash(newNode)] = len(self.queue) - 1

        # Bubble up the new node
        self.bubbleUp(newNode)

        if self.queueSizeLimit > 0 and len(self.queue) > self.queueSizeLimit:
            delNode = self.queue[-1]
            self.dic.pop(hash(self.queue[-1]))
            self.queue.__delitem__(-1)

        return newNode

    # Removes the root, and returns its element
    def deleteMin(self):
        if len(self.queue) < 1:
            return

        # Store the root node in temp.
        root = self.queue[0]

        # Swap the last node with the root
        if len(self.queue) > 1:
            self.swap(self.queue[0], self.queue[len(self.queue) - 1])

        # Remove the root from its new parent
        if root in root.parent.children:
            root.parent.children.remove(root)

        # Remove the node from the dictionary
        self.dic.pop(hash(root))

        # Remove the node from the queue
        self.queue.__delitem__(len(self.queue) - 1)

        # Pulls out the node's element
        temp = root.delete()

        # Let the new root bubble back down
        if self.notEmpty():
            self.bubbleDown(self.queue[0])

        # Returns the element stored inside the old root
        return temp

    # Verifies that the node is in the correct position in the priority queue. Use after lowering its priority.
    def decreaseKey(self, modifiedNode):
        self.bubbleUp(modifiedNode)
        return modifiedNode

    # Updates the priority of the node
    def changePriority(self, nodeToModify, priority):
        oldPriority = nodeToModify.priority
        nodeToModify.priority = priority
        if oldPriority > priority:
            self.decreaseKey(nodeToModify)
        else:
            self.bubbleDown(nodeToModify)
        return nodeToModify

    # Swaps two nodes.
    def swap(self, node1, node2):
        if type(node1) == int:
            node1 = self.queue[node1]
        if type(node2) == int:
            node2 = self.queue[node2]
        n1Index = self.dic[hash(node1)]
        n2Index = self.dic[hash(node2)]
        node1Parent = node1.parent
        node2Parent = node2.parent

        # Swap the nodes in the queue array
        self.queue[n1Index], self.queue[n2Index] = self.queue[n2Index], self.queue[n1Index]

        # Update the node pointers in the dictionary
        self.dic[hash(node1)] = n2Index
        self.dic[hash(node2)] = n1Index

        # Swap the nodes' children
        node1.children, node2.children = node2.children, node1.children

        # Assign the nodes' parents
        # If node2 is node1's parent:
        if node1.parent == node2:
            # If node2 is the root:
            if node2.parent == node2:
                node1.parent = node1
            # Else node2 is mid-tree
            else:
                node1.parent = node2.parent
            node2.parent = node1

        # If node1 is node2's parent:
        else:
            if node2.parent == node1:
                # If node1 is the root:
                if node1.parent == node1:
                    node2.parent = node2
                # Else node1 is mid-tree
                else:
                    node2.parent = node1.parent
                node1.parent = node2

            # If neither node is the other's parent:
            else:
                # If node1 is the root:
                if node1.parent == node1:
                    node1.parent = node2.parent
                    node2.parent = node2
                # If node2 is the root:
                else:
                    if n2Index == 0:
                        node2.parent = node1.parent
                        node1.parent = node1
                    # If neither node is the parent of the other, and neither is the root:
                    else:
                        node1.parent, node2.parent = node2.parent, node1.parent

        # Add the nodes to their new parent's children list
        if node1.parent != node1:
            node1.parent.children.append(node1)
        if node2.parent != node2:
            node2.parent.children.append(node2)

        # Remove the nodes from old parent's children list
        if node2Parent != node1 and node2 in node2Parent.children:
            node2Parent.children.remove(node2)
        if node1Parent != node2 and node1 in node1Parent.children:
            node1Parent.children.remove(node1)

        # Remove the nodes if they were added to their own children during the children swap:
        if node1 in node1.children:
            node1.children.remove(node1)
        if node2 in node2.children:
            node2.children.remove(node2)

        # Reassign the nodes to be their new children's parents
        for c in node1.children:
            c.parent = node1
        for c in node2.children:
            c.parent = node2

    # Moves the node down the priority queue, further from the root, while its children have lower edge weights/priority
    def bubbleDown(self, nodeToSettle):
        # Swap node and its child if its child has lower priority (lower edge weight)
        childToSwap = None
        for c in nodeToSettle.children:
            if nodeToSettle.priority > c.priority:
                if childToSwap is None or childToSwap.priority > c.priority:
                    childToSwap = c
        if childToSwap is not None:
            self.swap(nodeToSettle, childToSwap)
            self.bubbleDown(nodeToSettle)
        return nodeToSettle

    # Moves the node up the priority queue, closer to the root, while its parent has a higher edge weight/priority
    def bubbleUp(self, nodeToSettle):
        # Swap node and its child if its child has lower priority (lower edge weight)
        if nodeToSettle.parent.priority > nodeToSettle.priority:
            self.swap(nodeToSettle, nodeToSettle.parent)
            self.bubbleUp(nodeToSettle)
        return nodeToSettle

    def notEmpty(self):
        return super().notEmpty()

