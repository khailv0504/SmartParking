import unittest
import numpy as np
from edge.recognition import Box, group_text_rows,letterbox, OCR_CLASSES

class OCRTest(unittest.TestCase):
    def make(self,s,y):
        return [Box(i*15,y,i*15+10,y+18,.9,OCR_CLASSES.index(ch))
                for i,ch in enumerate(s)]
    def test_single_line(self):
        boxes=self.make('59A12345',10)
        self.assertEqual(group_text_rows(boxes),('59A12345',))
    def test_two_lines(self):
        boxes=self.make('59A',5)+self.make('12345',35)
        self.assertEqual(group_text_rows(boxes),('59A','12345'))
    def test_letterbox_aspect(self):
        result,scale,left,top=letterbox(np.zeros((720,1280,3),np.uint8),640)
        self.assertEqual(result.shape,(640,640,3))
        self.assertEqual(scale,.5)
        self.assertEqual((left,top),(0,140))

if __name__=='__main__': unittest.main()
