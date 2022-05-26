from random import randrange

class DecisionTreeClassifier():
	def __init__(self, max_depth, min_size):
		self.max_depth = max_depth
		self.min_size = min_size
	
	# Разделение данных по критерию (условие if-else)
	def test_split(self, index, value, dataset):
		left, right = list(), list()
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right # возвращаем левого и правого потомков после разделения
	
	# Расчет индекса Джини для оптимального разделения
	def gini_index(self, groups, classes):
		# подсчет всех элементов в точке разделения
		n_instances = float(sum([len(group) for group in groups]))
		# индекс Джини для группы
		gini = 0.0

		for group in groups:
			size = float(len(group))
			# пропускаем пустую подгруппу
			if size == 0:
				continue

			score = 0.0

			# сумма квадратов вероятностей классов в подгруппе
			for class_val in classes:
				p = [row[-1] for row in group].count(class_val) / size
				score += p * p

			# сумма всвешенных индексов Джини для всех потомков 
			# (домноженных на долю элементов подгруппы относительно изначальной группы)
			gini += (1.0 - score) * (size / n_instances)

		return gini
	
	# Выбор лучшего критерия разделения (с наименьшим значением индекса Джини)
	def get_split(self, dataset, n_features):
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		features = list()
		while len(features) < n_features:
			index = randrange(len(dataset[0])-1)
			if index not in features:
				features.append(index)
		for index in features:
			for row in dataset:
				groups = self.test_split(index, row[index], dataset)
				gini = self.gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		return {'index':b_index, 'value':b_value, 'groups':b_groups}
	
	# Создание конечного узла
	def to_terminal(self, group):
		outcomes = [row[-1] for row in group] # классы, к которым относятся элементы группы
		return max(set(outcomes), key=outcomes.count) # присвоение узлу того класса, в котором больше элементов из группы
	
	# Добавление узлов-потомков для узла или преобразование узла в конечный (рекурсионно)
	def split(self, node, n_features, depth):
		left, right = node['groups']
		del(node['groups']) # удаляем данные до разделения

		# если узел не разделен, делаем узел конечным
		if not left or not right:
			node['left'] = node['right'] = self.to_terminal(left + right)
			return
		# если вышли за рамки макс. глубины, делаем узел конечным
		if depth >= self.max_depth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
			return

		# обработка левого потомка
		if len(left) <= self.min_size:
			node['left'] = self.to_terminal(left)
		else:
			node['left'] = self.get_split(left, n_features)
			self.split(node['left'], n_features, depth+1)

		# обработка правого потомка
		if len(right) <= self.min_size:
			node['right'] = self.to_terminal(right)
		else:
			node['right'] = self.get_split(right, n_features)
			self.split(node['right'], n_features, depth+1)
	
	# Построение дерева решений
	def build_tree(self, train, n_features):
		root = self.get_split(train, n_features)
		self.split(root, n_features, 1)
		return root # возвращаем полученное из корня дерево
	
	# Предсказание после обучения модели (рекурсионно)
	@staticmethod
	def predict(node, row):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return DecisionTreeClassifier.predict(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return DecisionTreeClassifier.predict(node['right'], row)
			else:
				return node['right']