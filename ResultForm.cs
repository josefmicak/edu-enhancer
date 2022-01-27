using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;

namespace TAO_Enhancer
{
    public partial class ResultForm : Form
    {
        string studentIdentifier = "";
        public ResultForm(string studentID)
        {
            InitializeComponent();
            studentIdentifier = studentID;
            LoadResults();
        }

        public void LoadResults()
        {
            int amountOfTests = 0;

            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\results"))
            {
                foreach (var file in Directory.GetFiles(directory))
                {
                    string extension = Path.GetExtension(file);
                    if (extension == ".xml")
                    {
                        bool addTest = false;
                        string timeStamp = "";
                        string testStudentIdentifier = "";

                        XmlReader xmlReader = XmlReader.Create(file);
                        while (xmlReader.Read())
                        {
                            if (xmlReader.Name == "context")
                            {
                                testStudentIdentifier = xmlReader.GetAttribute("sourcedId");
                                if (testStudentIdentifier == studentIdentifier)
                                {
                                    addTest = true;
                                }
                            }

                            if (xmlReader.Name == "testResult" && xmlReader.GetAttribute("datestamp") != null)
                            {
                                timeStamp = xmlReader.GetAttribute("datestamp");
                            }
                        }

                        if(addTest)
                        {
                            ResultsGridView.Rows.Add();
                            ResultsGridView.Rows[amountOfTests].Cells[0].Value = Path.GetFileName(directory).ToString();
                            ResultsGridView.Rows[amountOfTests].Cells[1].Value = timeStamp;
                            amountOfTests++;
                        }
                    }
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new StudentForm().Show();
            Hide();
        }
    }
}
