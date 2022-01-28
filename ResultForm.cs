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
using VDS.RDF;
using VDS.RDF.Parsing;

namespace TAO_Enhancer
{
    public partial class ResultForm : Form
    {
        string studentIdentifier = "";
        int selectedAttempt = -1;
        List<string> attemptIdentifier = new List<string>();

        public ResultForm(string studentID)
        {
            InitializeComponent();
            studentIdentifier = studentID;
            LoadStudent();
            LoadResults();
        }

        public void LoadStudent()
        {
            string login = "", name = "", surname = "", email = "";

            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\testtakers"))
            {
                string extension = Path.GetExtension(file);
                if (extension == ".rdf")
                {
                    IGraph g = new Graph();
                    FileLoader.Load(g, file);
                    IEnumerable<INode> nodes = g.AllNodes;
                    int nodeLine = 1;//TODO: viz TODO 1
                    foreach (INode node in nodes)
                    {
                        if (nodeLine == 1)
                        {
                            string[] splitByHashtag = node.ToString().Split("#");
                        }
                        if (nodeLine == 3)
                        {
                            login = node.ToString();
                        }
                        else if (nodeLine == 9)
                        {
                            name = node.ToString();
                        }
                        else if (nodeLine == 11)
                        {
                            surname = node.ToString();
                        }
                        else if (nodeLine == 13)
                        {
                            email = node.ToString();
                        }
                        nodeLine++;
                    }
                }
            }

            StudentNameLabel.Text = "Jméno: " + name + " " + surname;
            StudentLoginLabel.Text = "Login: " + login;
            StudentEmailLabel.Text = "Email: " + email;
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

                        string[] attemptIdentifierSplitByUnderscore = Path.GetFileNameWithoutExtension(file).Split("_");

                        if(addTest)
                        {
                            ResultsGridView.Rows.Add();
                            ResultsGridView.Rows[amountOfTests].Cells[0].Value = Path.GetFileName(directory).ToString();
                            ResultsGridView.Rows[amountOfTests].Cells[1].Value = timeStamp;
                            ResultsGridView.Rows[amountOfTests].Cells[2].Value = attemptIdentifierSplitByUnderscore[2];
                            attemptIdentifier.Add(attemptIdentifierSplitByUnderscore[2]);
                            amountOfTests++;
                        }
                    }
                }
            }

            AmountOfTestsLabel.Text = "Počet vyplněných testů: " + amountOfTests;
        }

        public (string, string) GetTestIdentifiers()
        {
            string testNameIdentifier = ResultsGridView.Rows[selectedAttempt].Cells[0].Value.ToString();
            string testNumberIdentifier = "";
            foreach (var directory in Directory.GetDirectories("C:\\xampp\\exported\\tests\\" + testNameIdentifier + "\\tests"))
            {
                testNumberIdentifier = Path.GetFileName(directory);
            }
            return (testNameIdentifier, testNumberIdentifier);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new StudentForm().Show();
            Hide();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (selectedAttempt == -1)
            {
                MessageBox.Show("Chyba - nevybral jste žadný výsledek testu. Prosím vyberte výsledek testu kliknutím na příslušný řádek.", "Nebyl vybrán výsledek testu", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                new TestForm(GetTestIdentifiers(), false, attemptIdentifier[selectedAttempt], studentIdentifier).Show();
                Hide();
            }
            
        }

        private void ResultsGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            selectedAttempt = ResultsGridView.CurrentCell.RowIndex;
        }
    }
}
