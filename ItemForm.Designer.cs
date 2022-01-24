
namespace TAO_Enhancer
{
    partial class ItemForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.IdentifierLabel = new System.Windows.Forms.Label();
            this.TitleLabel = new System.Windows.Forms.Label();
            this.LabelLabel = new System.Windows.Forms.Label();
            this.QuestionLabel = new System.Windows.Forms.Label();
            this.CorrectAnswerLabel = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.QuestionImage = new System.Windows.Forms.PictureBox();
            this.PossibleAnswerLabel = new System.Windows.Forms.Label();
            this.SubquestionLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.QuestionImage)).BeginInit();
            this.SuspendLayout();
            // 
            // IdentifierLabel
            // 
            this.IdentifierLabel.AutoSize = true;
            this.IdentifierLabel.Location = new System.Drawing.Point(29, 54);
            this.IdentifierLabel.Name = "IdentifierLabel";
            this.IdentifierLabel.Size = new System.Drawing.Size(82, 15);
            this.IdentifierLabel.TabIndex = 0;
            this.IdentifierLabel.Text = "IdentifierLabel";
            // 
            // TitleLabel
            // 
            this.TitleLabel.AutoSize = true;
            this.TitleLabel.Location = new System.Drawing.Point(29, 92);
            this.TitleLabel.Name = "TitleLabel";
            this.TitleLabel.Size = new System.Drawing.Size(57, 15);
            this.TitleLabel.TabIndex = 1;
            this.TitleLabel.Text = "TitleLabel";
            // 
            // LabelLabel
            // 
            this.LabelLabel.AutoSize = true;
            this.LabelLabel.Location = new System.Drawing.Point(29, 127);
            this.LabelLabel.Name = "LabelLabel";
            this.LabelLabel.Size = new System.Drawing.Size(63, 15);
            this.LabelLabel.TabIndex = 2;
            this.LabelLabel.Text = "LabelLabel";
            // 
            // QuestionLabel
            // 
            this.QuestionLabel.AutoSize = true;
            this.QuestionLabel.Location = new System.Drawing.Point(476, 255);
            this.QuestionLabel.Name = "QuestionLabel";
            this.QuestionLabel.Size = new System.Drawing.Size(83, 15);
            this.QuestionLabel.TabIndex = 3;
            this.QuestionLabel.Text = "QuestionLabel";
            // 
            // CorrectAnswerLabel
            // 
            this.CorrectAnswerLabel.AutoSize = true;
            this.CorrectAnswerLabel.Location = new System.Drawing.Point(758, 304);
            this.CorrectAnswerLabel.Name = "CorrectAnswerLabel";
            this.CorrectAnswerLabel.Size = new System.Drawing.Size(113, 15);
            this.CorrectAnswerLabel.TabIndex = 4;
            this.CorrectAnswerLabel.Text = "CorrectAnswerLabel";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(552, 693);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(75, 23);
            this.button1.TabIndex = 5;
            this.button1.Text = "Zpět";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // QuestionImage
            // 
            this.QuestionImage.Location = new System.Drawing.Point(476, 40);
            this.QuestionImage.Name = "QuestionImage";
            this.QuestionImage.Size = new System.Drawing.Size(395, 198);
            this.QuestionImage.TabIndex = 6;
            this.QuestionImage.TabStop = false;
            // 
            // PossibleAnswerLabel
            // 
            this.PossibleAnswerLabel.AutoSize = true;
            this.PossibleAnswerLabel.Location = new System.Drawing.Point(476, 304);
            this.PossibleAnswerLabel.Name = "PossibleAnswerLabel";
            this.PossibleAnswerLabel.Size = new System.Drawing.Size(117, 15);
            this.PossibleAnswerLabel.TabIndex = 7;
            this.PossibleAnswerLabel.Text = "PossibleAnswerLabel";
            // 
            // SubquestionLabel
            // 
            this.SubquestionLabel.AutoSize = true;
            this.SubquestionLabel.Location = new System.Drawing.Point(476, 274);
            this.SubquestionLabel.Name = "SubquestionLabel";
            this.SubquestionLabel.Size = new System.Drawing.Size(101, 15);
            this.SubquestionLabel.TabIndex = 8;
            this.SubquestionLabel.Text = "SubquestionLabel";
            this.SubquestionLabel.Visible = false;
            // 
            // ItemForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1223, 742);
            this.Controls.Add(this.SubquestionLabel);
            this.Controls.Add(this.PossibleAnswerLabel);
            this.Controls.Add(this.QuestionImage);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.CorrectAnswerLabel);
            this.Controls.Add(this.QuestionLabel);
            this.Controls.Add(this.LabelLabel);
            this.Controls.Add(this.TitleLabel);
            this.Controls.Add(this.IdentifierLabel);
            this.Name = "ItemForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "ItemForm";
            ((System.ComponentModel.ISupportInitialize)(this.QuestionImage)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label IdentifierLabel;
        private System.Windows.Forms.Label TitleLabel;
        private System.Windows.Forms.Label LabelLabel;
        private System.Windows.Forms.Label QuestionLabel;
        private System.Windows.Forms.Label CorrectAnswerLabel;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.PictureBox QuestionImage;
        private System.Windows.Forms.Label PossibleAnswerLabel;
        private System.Windows.Forms.Label SubquestionLabel;
    }
}