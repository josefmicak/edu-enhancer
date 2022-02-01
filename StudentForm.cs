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
using VDS.RDF;
using VDS.RDF.Parsing;
using VDS.RDF.Query;

namespace TAO_Enhancer
{
    public partial class StudentForm : Form
    {
        List<string> studentIdentifier = new List<string>();
        int selectedStudent = -1;
        public StudentForm()
        {
            InitializeComponent();
            LoadTestTakers();
        }

        public void LoadTestTakers()
        {
            int gridViewRow = 0;
            foreach (var file in Directory.GetFiles("C:\\xampp\\exported\\testtakers"))
            {
                string extension = Path.GetExtension(file);
                if(extension == ".rdf")
                {
                    IGraph g = new Graph();
                    FileLoader.Load(g, file);
                    IEnumerable<INode> nodes = g.AllNodes;
                    int nodeLine = 1;//TODO 1: Předělat; Udělat podmínky jako if(node == ns0:userFirstName)
                    string login = "", name = "", surname = "";
                    foreach (INode node in nodes)
                    {
                        if (nodeLine == 1)
                        {
                            string[] splitByHashtag = node.ToString().Split("#");
                            studentIdentifier.Add(splitByHashtag[1]);
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
                        nodeLine++;
                    }
                    TestTakersGridView.Rows.Add();
                    TestTakersGridView.Rows[gridViewRow].Cells[0].Value = login;
                    TestTakersGridView.Rows[gridViewRow].Cells[1].Value = name + " " + surname;
                    gridViewRow++;
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new EntryForm().Show();
            Hide();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (selectedStudent == -1)
            {
                MessageBox.Show("Chyba - nevybral jste žadného studenta. Prosím vyberte studenta kliknutím na příslušný řádek.", "Nebyl vybrán student", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                new ResultForm(studentIdentifier[selectedStudent], false).Show();
                Hide();
            }
        }

        private void TestTakersGridView_CellClick(object sender, DataGridViewCellEventArgs e)
        {
            selectedStudent = TestTakersGridView.CurrentCell.RowIndex;
        }
    }
}
