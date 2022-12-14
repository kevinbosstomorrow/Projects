
from manim import *
# manim -pqh parameters_shapes.py Conv1d_example1


# the size of one single matrix, the number of vector matrix, the number of matrix (一个单独的方块， 一排， n个一排)
def get_rectangle(scalar_height, scalar_width, vector_size, matrix_size, stroke_size, col):
    scaler = VGroup(*[Rectangle(height = scalar_height, width = scalar_width, stroke_width = stroke_size).set_color(col) for _ in range(vector_size)]).arrange(RIGHT, buff=0)
    matrix = VGroup(*[scaler.copy() for _ in range(matrix_size)]).arrange(DOWN, buff=0)

    return matrix

class Conv1d_example2(Scene):
    def construct(self):
        self.camera.background_color = GREY_E

        #df = get_rectangle(1,1,10,2,0.5,WHITE).center().scale(0.5)
        df1 = get_rectangle(1,3,10,2,0.5,WHITE).center()

        time = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6','Day7','Day8','Day9','Day10']
        value = ['12', '23', '33', '53', '36', '40','45','55','46','56']

        for i,t in enumerate(time):
            df1.add(Text(t, color= WHITE).scale(0.7).move_to(df1[0][i].get_center()))

        for i,t in enumerate(value):            
            df1.add(Text(t, color= WHITE).scale(0.7).move_to(df1[1][i].get_center()))
        df1.scale(0.3)
        self.play(FadeIn(df1))
        self.wait(2)
        
        # label2 - features 
        label2 = VGroup(*[Rectangle(height = 1, width = 3, stroke_width = 0.5).set_color(YELLOW).set_opacity(0.2) for _ in range(9)]).arrange(RIGHT, buff=0)
        label2 = VGroup(*[label2.copy() for _ in range(2)]).arrange(DOWN, buff=0)
        label2.scale(0.3).shift(LEFT*0.45)
        self.play(FadeIn(label2))
        self.wait(2)

        #label3 = trade this like label 
        label3 = VGroup(*[Rectangle(height = 1, width = 3, stroke_width = 0.5).set_color(BLUE).set_opacity(0.2) for _ in range(1)]).arrange(RIGHT, buff=0)
        label3 = VGroup(*[label3.copy() for _ in range(2)]).arrange(DOWN, buff=0)
        label3.scale(0.3).shift(RIGHT*4.05)        

        self.play(Transform(label2, label3), run_time = 2)
        self.wait(2)

        # 括号 brace 
        x_axis = VGroup()
        x_axis.add(df1[0][0], df1[1][0])

        brace_x_axis = Brace(x_axis, sharpness=1).move_to(x_axis.get_left()).rotate(4.7).shift(LEFT*0.2).scale(0.7)
        brace_y_axis = Brace(df1, sharpness=1)


        # 数字 shape 1 and 10 
        num1 = Text('1').move_to(brace_x_axis.get_left()).scale(0.7).shift(LEFT*0.2)
        num2 = Text('10').move_to(brace_y_axis.get_bottom()).scale(0.7).shift(DOWN*0.2)

        self.play(FadeIn(brace_x_axis), FadeIn(num1), FadeIn(brace_y_axis), FadeIn(num2))
        

        shapes = Text('shape (batch, timesteps, features)').scale(0.5).center().shift(DOWN*2)
        none = Text('None').move_to(shapes.get_bottom()).scale(0.5).shift(DOWN*0.5).shift(LEFT*1)
        self.play(FadeIn(shapes), FadeIn(none),
                num1.animate.move_to(shapes.get_bottom()).shift(DOWN*0.5).shift(RIGHT*0.5), 
                num2.animate.move_to(shapes.get_bottom()).shift(DOWN*0.5).shift(RIGHT*1.7), run_time = 2)
        self.wait(2)


        self.play(FadeOut(brace_x_axis),FadeOut(brace_y_axis),FadeOut(label2))
        self.play(df1.animate.shift(UP*3))
        self.wait(1)

        input_layer = VGroup()
        input_layer.add(shapes, none, num1, num2)


        self.play(input_layer.animate.shift(UP*3, LEFT*4).scale(0.6))


        param = Text('Param #').scale(0.3).move_to(input_layer[0].get_right()).shift(RIGHT*1)
        param1 = Text('0').scale(0.42).move_to(input_layer[3].get_right()).shift(RIGHT*1.2)
        self.play(FadeIn(param), FadeIn(param1))


        ''' Conv1D '''
        conv1d = VGroup(*[Rectangle(height = 1, width = 1, stroke_width = 2 ).set_color(RED).set_opacity(0.5) for _ in range(3)]).arrange(RIGHT, buff=0)
        conv1d.scale(0.3).move_to(df1[1][0].get_center())
        self.play(FadeIn(conv1d))

        conv1d2 = VGroup(*[Rectangle(height = 1, width = 1, stroke_width = 2 ).set_color(RED).set_opacity(0.5) for _ in range(30)]).arrange(RIGHT, buff=0)
        conv1d2.scale(0.3).move_to(df1[1].get_center())
    

        self.play(Transform(conv1d, conv1d2))
        self.wait(2)

        conv1d3 = VGroup(*[Rectangle(height = 1, width = 1, stroke_width = 2 ).set_color(RED).set_opacity(0.5) for _ in range(30)]).arrange(RIGHT, buff=0)
        conv1d3 = VGroup(*[conv1d3.copy() for _ in range(5)]).arrange(DOWN, buff=0)
        conv1d3.scale(0.3).move_to(df1[1].get_center()).shift(DOWN*1)
        self.play(Transform(conv1d, conv1d3))
        self.wait(2)

        bias = VGroup(*[Rectangle(height = 1, width = 1, stroke_width = 2 ).set_color(RED).set_opacity(0.5) for _ in range(5)]).arrange(RIGHT, buff=0).scale(0.3)
        bias.rotate(4.7).move_to(conv1d.get_right()).shift(RIGHT*1)
        self.play(FadeIn(bias))
        self.wait(2)

        # conv layer parameters 
        param2 = Text('3*10*5 + 5 = 155').scale(0.42).move_to(param1.get_bottom()).shift(DOWN*0.5).shift(RIGHT*1)
        self.play(Create(param2), run_time = 2)

        # shape 
        self.play(FadeOut(bias))
         
        shape_conv = VGroup(*[Rectangle(height = 1, width = 1, stroke_width = 2 ).set_color(RED).set_opacity(0.5) for _ in range(5)]).arrange(RIGHT, buff=0).scale(0.3).shift(UP*1, RIGHT*1) 
        self.play(conv1d.animate.rotate(4.705).shift(RIGHT*1))
        self.wait(2)
        self.play(Transform(conv1d, shape_conv), run_time = 1)
        self.wait(2)


        shapes2 = Text('None       1       5').scale(0.42).move_to(input_layer[2].get_bottom()).shift(DOWN*0.5,LEFT*0.3)
        self.play(Create(shapes2))
        self.wait(2)




class Conv1d_example1(Scene):
    def construct(self):
        self.camera.background_color = GREY_E

    text1 =Text('Kevinboss is awesome and pretty cool')
