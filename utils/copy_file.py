# import os
# import shutil
#
#
#
# def copyPDF():
#  addressCifar = "E:/totally/FinancePDF/"
#
#  f_list = os.listdir(addressCifar)
#  n=50
#  count=0
#  for fileNAME in f_list:
#     count=count+1
#     if count == n:
#         break
#     if os.path.splitext(fileNAME)[1] == '.pdf':
#         n += 1
#         oldname = u"E:\\totally\\FinancePDF\\" + fileNAME
#         newname = u"E:\\totally\\Finance_PDFfile\\" + fileNAME
#         shutil.copyfile(oldname, newname)
#         print(str(n)+'.'+'已复制'+fileNAME)
#
# if __name__ == '__main__':
#     copyPDF()