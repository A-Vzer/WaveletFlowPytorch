from nf.layers import SqueezeLayer

class Conditioning_network:
	def __init__(self):
		self.encoder_list = [
			None,
			self.encode_1,
			self.encode_2,
			self.encode_3,
			self.encode_4,
			self.encode_5,
			self.encode_6,
			self.encode_7,
			self.encode_8,
			self.encode_9,
			self.encode_10,
		]
		self.squeeze_layer = SqueezeLayer(factor=2)

	def encode(self,downsampled_stack):
		'''
		these networks must be manually specified,
		not setup procedurally via config
		'''
		conditioning_tensors = []
		for it,encoder in enumerate(self.encoder_list):
			conditioning_tensors.append(encoder(downsampled_stack[it]))

		return conditioning_tensors

	def encode_1(self,base):
		return base

	def encode_2(self,base):
		return base

	def encode_3(self,base):
		return base

	def encode_4(self,base):
		return base

	def encode_5(self,base):
		return base

	def encode_6(self,base):
		return base

	def encode_7(self, base):
		return base

	def encode_8(self,base):
		return base

	def encode_9(self,base):
		return base

	def encode_10(self,base):
		return base