"""
Manimを使ったニューラルネットワークの予測過程の視覚化
"""
import torch
import torch.nn as nn
import numpy as np
from manim import *
from models.simple_cnn import Simple_CNN
from data import get_data_loaders
from typing import Dict

MODEL_PATH = "models/simple_cnn.pth"


class NNVisualizer:
    """ニューラルネットワークの中間層出力を取得するクラス"""

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Simple_CNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        # 中間層の出力を保存する辞書
        self.activations: Dict[str, torch.Tensor] = {}

    def get_activation(self, name: str):
        """指定された層の出力を保存するフック関数"""
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        """各層にフックを登録"""
        self.model.layer1.register_forward_hook(self.get_activation('layer1'))
        self.model.layer2.register_forward_hook(self.get_activation('layer2'))
        self.model.layer3.register_forward_hook(self.get_activation('layer3'))
        self.model.layer4.register_forward_hook(self.get_activation('layer4'))

    def predict_with_activations(self, image: torch.Tensor):
        """
        画像を入力して予測結果と各層の活性化を取得

        Args:
            image: (1, 1, 28, 28) の入力画像テンソル

        Returns:
            予測結果と各層の活性化を含む辞書
        """
        self.activations.clear()
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        return {
            'input': image.cpu(),
            'layer1_conv': self.activations['layer1'].cpu(),  # (1, 16, 13, 13)
            'layer2_conv': self.activations['layer2'].cpu(),  # (1, 32, 5, 5)
            'layer3_fc': self.activations['layer3'].cpu(),    # (1, 128)
            'output': output.cpu(),                            # (1, 10)
            'probabilities': probabilities.cpu(),
            'predicted_class': predicted_class.item()
        }


class NeuralNetworkVisualization(Scene):
    """Manimシーン: ニューラルネットワークの予測過程を視覚化"""

    def construct(self):
        # データとモデルの準備
        _, test_loader = get_data_loaders(batch_size=1)
        visualizer = NNVisualizer(MODEL_PATH)
        visualizer.register_hooks()

        # メインタイトル
        main_title = Text("Neural Network Prediction Visualization", font_size=36)
        main_title.to_edge(UP)
        self.play(Write(main_title))
        self.wait(1)
        self.play(FadeOut(main_title))

        # 複数の画像を処理（5枚）
        num_images = 5
        test_iter = iter(test_loader)

        # ランダムな開始位置
        skip_count = np.random.randint(0, 100)
        for _ in range(skip_count):
            next(test_iter)

        for img_idx in range(num_images):
            try:
                image, label = next(test_iter)
            except StopIteration:
                break

            # 予測と活性化の取得
            result = visualizer.predict_with_activations(image)
            true_label = label.item()

            # 画像番号を表示
            image_counter = Text(
                f"Image {img_idx + 1}/{num_images}",
                font_size=28,
                color=YELLOW
            )
            image_counter.to_edge(UP)
            self.play(Write(image_counter))
            self.wait(0.5)

            # 1. 入力画像の表示
            input_objects = self.show_input_image(result['input'], true_label, image_counter)
            self.wait(1)
            self.play(FadeOut(input_objects))

            # 2. Layer1 (Conv1) の畳み込み層視覚化
            layer1_objects = self.show_conv_layer(
                result['layer1_conv'],
                layer_name="Layer 1: Conv2d (16 filters)",
                position=UP * 2,
                counter=image_counter
            )
            self.wait(1)
            self.play(FadeOut(layer1_objects))

            # 3. Layer2 (Conv2) の畳み込み層視覚化
            layer2_objects = self.show_conv_layer(
                result['layer2_conv'],
                layer_name="Layer 2: Conv2d (32 filters)",
                position=ORIGIN,
                counter=image_counter
            )
            self.wait(1)
            self.play(FadeOut(layer2_objects))

            # 4. Layer3 (FC1) の全結合層視覚化
            layer3_objects = self.show_fc_layer(
                result['layer3_fc'],
                layer_name="Layer 3: Fully Connected (128 neurons)",
                position=DOWN * 1.5,
                counter=image_counter
            )
            self.wait(1)
            self.play(FadeOut(layer3_objects))

            # 5. 出力層の視覚化
            output_objects = self.show_output_layer(
                result['probabilities'],
                result['predicted_class'],
                true_label,
                counter=image_counter
            )
            self.wait(2)

            # 次の画像へ移行（最後の画像でない場合）
            if img_idx < num_images - 1:
                self.play(FadeOut(output_objects), FadeOut(image_counter))
                self.wait(0.5)
            else:
                self.play(FadeOut(output_objects), FadeOut(image_counter))
                self.wait(1)

        # エンディング
        ending_text = Text("Visualization Complete", font_size=40, color=GREEN)
        self.play(Write(ending_text))
        self.wait(2)

    def show_input_image(self, image_tensor: torch.Tensor, true_label: int, counter=None):
        """入力画像を表示"""
        # タイトル
        title = Text("Input Image", font_size=32)
        if counter:
            title.next_to(counter, DOWN, buff=0.3)
        else:
            title.to_edge(UP)

        # 画像データを取得 (正規化を戻す)
        img_data = image_tensor[0, 0].numpy()
        # MNIST の正規化を戻す: mean=0.1307, std=0.3081
        img_data = img_data * 0.3081 + 0.1307
        img_data = np.clip(img_data, 0, 1)

        # 画像を28x28のピクセルとして表示
        pixel_size = 0.15
        pixels = VGroup()

        for i in range(28):
            for j in range(28):
                intensity = img_data[i, j]
                pixel = Square(side_length=pixel_size)
                pixel.set_fill(WHITE, opacity=intensity)
                pixel.set_stroke(GRAY, width=0.5)
                pixel.move_to(np.array([
                    (j - 14) * pixel_size,
                    (14 - i) * pixel_size,
                    0
                ]))
                pixels.add(pixel)

        pixels.scale(0.8)

        # ラベル情報
        label_text = Text(f"True Label: {true_label}", font_size=24)
        label_text.next_to(pixels, DOWN, buff=0.5)

        # テンソル形状の表示
        shape_text = Text(f"Shape: (1, 1, 28, 28)", font_size=20, color=BLUE)
        shape_text.next_to(label_text, DOWN, buff=0.3)

        self.play(Write(title))
        self.play(Create(pixels), run_time=2)
        self.play(Write(label_text), Write(shape_text))
        self.wait(2)

        # すべてのオブジェクトをグループ化して返す
        return VGroup(title, pixels, label_text, shape_text)

    def show_conv_layer(self, feature_maps: torch.Tensor, layer_name: str, position, counter=None):
        """畳み込み層の特徴マップを表示"""
        title = Text(layer_name, font_size=28)
        if counter:
            title.next_to(counter, DOWN, buff=0.3)
        else:
            title.to_edge(UP)

        # feature_maps shape: (1, channels, H, W)
        feature_maps = feature_maps[0].numpy()  # (channels, H, W)
        num_channels, height, width = feature_maps.shape

        # 形状情報
        shape_text = Text(
            f"Shape: (1, {num_channels}, {height}, {width})",
            font_size=20,
            color=BLUE
        )
        shape_text.next_to(title, DOWN, buff=0.2)

        # 特徴マップをグリッド表示
        # レイアウト: できるだけ正方形に近い配置
        cols = int(np.ceil(np.sqrt(num_channels)))
        rows = int(np.ceil(num_channels / cols))

        feature_map_size = 0.8 if num_channels <= 16 else 0.5
        spacing = 0.2

        feature_maps_group = VGroup()

        for idx in range(num_channels):
            fm = feature_maps[idx]
            # 正規化
            fm_min, fm_max = fm.min(), fm.max()
            if fm_max > fm_min:
                fm_normalized = (fm - fm_min) / (fm_max - fm_min)
            else:
                fm_normalized = np.zeros_like(fm)

            # ピクセルグリッドを作成
            pixels = VGroup()
            pixel_size = feature_map_size / max(height, width)

            for i in range(height):
                for j in range(width):
                    intensity = fm_normalized[i, j]
                    pixel = Square(side_length=pixel_size)
                    # 活性化度を色で表現（青→白→赤）
                    color = interpolate_color(BLUE, RED, intensity)
                    pixel.set_fill(color, opacity=0.8)
                    pixel.set_stroke(GRAY, width=0.1)
                    pixel.move_to(np.array([
                        (j - width/2) * pixel_size,
                        (height/2 - i) * pixel_size,
                        0
                    ]))
                    pixels.add(pixel)

            # グリッド配置
            row = idx // cols
            col = idx % cols

            # 中心からの配置
            x_offset = (col - cols/2 + 0.5) * (feature_map_size + spacing)
            y_offset = (rows/2 - row - 0.5) * (feature_map_size + spacing)

            pixels.move_to(np.array([x_offset, y_offset - 1, 0]))

            # チャンネル番号
            channel_label = Text(f"{idx}", font_size=12, color=YELLOW)
            channel_label.next_to(pixels, UP, buff=0.05)

            feature_maps_group.add(pixels, channel_label)

        self.play(Write(title), Write(shape_text))
        self.play(
            Create(feature_maps_group),
            run_time=3,
            rate_func=smooth
        )
        self.wait(2)

        # すべてのオブジェクトをグループ化して返す
        return VGroup(title, shape_text, feature_maps_group)

    def show_fc_layer(self, activations: torch.Tensor, layer_name: str, position, counter=None):
        """全結合層のニューロンを視覚化"""
        title = Text(layer_name, font_size=28)
        if counter:
            title.next_to(counter, DOWN, buff=0.3)
        else:
            title.to_edge(UP)

        # activations shape: (1, 128)
        activations = activations[0].numpy()  # (128,)
        num_neurons = len(activations)

        # 正規化
        act_min, act_max = activations.min(), activations.max()
        if act_max > act_min:
            normalized_acts = (activations - act_min) / (act_max - act_min)
        else:
            normalized_acts = np.zeros_like(activations)

        # 形状情報
        shape_text = Text(
            f"Shape: (1, {num_neurons})",
            font_size=20,
            color=BLUE
        )
        shape_text.next_to(title, DOWN, buff=0.2)

        # ニューロンを円で表示（グリッド配置）
        cols = 16  # 16 x 8 = 128
        rows = 8

        neuron_size = 0.15
        spacing_x = 0.4
        spacing_y = 0.4

        neurons_group = VGroup()

        for idx in range(num_neurons):
            intensity = normalized_acts[idx]

            # 円を作成
            neuron = Circle(radius=neuron_size)

            # 活性化度を色と不透明度で表現
            color = interpolate_color(BLUE, RED, intensity)
            neuron.set_fill(color, opacity=0.3 + 0.7 * intensity)
            neuron.set_stroke(WHITE, width=1)

            # グリッド配置
            row = idx // cols
            col = idx % cols

            x_offset = (col - cols/2 + 0.5) * spacing_x
            y_offset = (rows/2 - row - 0.5) * spacing_y

            neuron.move_to(np.array([x_offset, y_offset - 1, 0]))

            neurons_group.add(neuron)

        # 活性化の凡例
        legend_text = Text(
            "Color intensity = Neuron activation",
            font_size=16,
            color=GRAY
        )
        legend_text.to_edge(DOWN)

        self.play(Write(title), Write(shape_text))
        self.play(
            Create(neurons_group),
            run_time=3,
            rate_func=smooth
        )
        self.play(Write(legend_text))
        self.wait(2)

        # すべてのオブジェクトをグループ化して返す
        return VGroup(title, shape_text, neurons_group, legend_text)

    def show_output_layer(self, probabilities: torch.Tensor, predicted: int, true_label: int, counter=None):
        """出力層（10クラスの確率）を視覚化"""
        title = Text("Output Layer: Class Probabilities", font_size=28)
        if counter:
            title.next_to(counter, DOWN, buff=0.3)
        else:
            title.to_edge(UP)

        # probabilities shape: (1, 10)
        probs = probabilities[0].numpy()  # (10,)

        # 棒グラフで確率を表示
        bars = VGroup()
        labels = VGroup()

        bar_width = 0.5
        max_bar_height = 4
        spacing = 0.7

        for i in range(10):
            prob = probs[i]
            bar_height = prob * max_bar_height

            # バーを作成
            bar = Rectangle(
                width=bar_width,
                height=bar_height,
                fill_opacity=0.8
            )

            # 色: 予測クラスは緑、正解ラベルは青、その他は灰色
            if i == predicted and i == true_label:
                bar.set_fill(GREEN)
                bar.set_stroke(GREEN, width=3)
            elif i == predicted:
                bar.set_fill(GREEN)
                bar.set_stroke(GREEN, width=3)
            elif i == true_label:
                bar.set_fill(BLUE)
                bar.set_stroke(BLUE, width=3)
            else:
                bar.set_fill(GRAY)
                bar.set_stroke(GRAY, width=1)

            # 位置調整
            x_pos = (i - 4.5) * spacing
            bar.move_to(np.array([x_pos, bar_height/2 - 1.5, 0]))

            # ラベル（クラス番号）
            class_label = Text(str(i), font_size=20)
            class_label.next_to(bar, DOWN, buff=0.1)

            # 確率値
            prob_label = Text(f"{prob:.2%}", font_size=14)
            prob_label.next_to(bar, UP, buff=0.1)

            bars.add(bar)
            labels.add(class_label, prob_label)

        # 凡例
        legend = VGroup()

        if predicted == true_label:
            legend_text = Text(
                f"Predicted: {predicted} (Correct!) - Confidence: {probs[predicted]:.2%}",
                font_size=20,
                color=GREEN
            )
        else:
            legend_text = Text(
                f"Predicted: {predicted}, True: {true_label} (Incorrect)",
                font_size=20,
                color=RED
            )

        legend_text.to_edge(DOWN)
        legend.add(legend_text)

        self.play(Write(title))
        self.play(
            Create(bars),
            Write(labels),
            run_time=2
        )
        self.play(Write(legend))
        self.wait(3)

        # すべてのオブジェクトをグループ化して返す
        return VGroup(title, bars, labels, legend)


def main():
    """
    動画をレンダリング
    実行コマンド:
    manim -pql visualize_manim.py NeuralNetworkVisualization

    品質オプション:
    -ql: 低品質（480p, 15fps）
    -qm: 中品質（720p, 30fps）
    -qh: 高品質（1080p, 60fps）
    -qk: 4K品質（2160p, 60fps）

    その他のオプション:
    -p: レンダリング後に動画を再生
    -s: 最後のフレームのみを画像として保存
    """
    pass


if __name__ == "__main__":
    main()
