using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TAO_Enhancer
{
    public partial class EntryForm : Form
    {
        public EntryForm()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new StudentForm().Show();
            Hide();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new TeacherForm().Show();
            Hide();
        }
    }
}
